"""Objective-conditioned, advisory guidance for behavioral rollouts.

The guidance in this module is deliberately separate from trial evaluation. It
helps a child policy understand scene geometry and choose actions, but the
authoritative result still comes from replaying raw actions against typed trial
objectives in :mod:`environment_generation.behavior_trial`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable

from .env_verification import (
    SceneObject3D,
    scene_object_at,
    select_scene_objects,
    spatial_relation_holds,
)
from .player import camera_relative_velocity
from .schema import EnvSpec3D
from .scene_geometry import (
    Placement2D,
    contact_approach_centers,
    footprint_signed_clearance,
    nearest_contact_center,
    nearest_interior_center,
    segment_intersects_footprint,
    support_radius_xy,
    volumes_overlap,
)


ROUTE_GRID_SIZE = 0.35
ROUTE_CLEARANCE_MARGIN = 0.12
ROUTE_MAX_CELLS = 12_000
NEAR_RISK_CLEARANCE = 0.65
INTERACTION_RANGE = 0.45

SUPPORT_TYPES = {"ground", "platform", "ramp"}
MOVABLE_TYPES = {"pushable_box", "ball", "cylinder"}
STATIC_BLOCKER_TYPES = {"wall", "static_box", "platform"}


@dataclass(frozen=True)
class _Obstacle:
    object_id: str
    kind: str
    obj: SceneObject3D
    clearance: float


def semantic_affordances(
    obj: SceneObject3D,
    *,
    mechanism_progress: float | None = None,
) -> list[str]:
    """Describe capabilities and risks without prescribing an objective."""

    values: list[str] = []
    semantic = obj.semantic_type
    if semantic in SUPPORT_TYPES:
        values.extend(["support_surface", "walkable_surface"])
    if semantic == "ramp":
        values.append("elevation_route")
    if semantic in {"wall", "static_box"}:
        values.extend(["static_blocker", "contactable"])
    if semantic in {"static_box", "platform"}:
        values.append("potential_climb_surface")
    if semantic in MOVABLE_TYPES or obj.body_type == "dynamic":
        values.extend(["movable", "pushable", "contactable"])
    if semantic in {"goal", "target_region"}:
        values.extend(["enterable_zone", "non_colliding_sensor"])
    if semantic == "hazard":
        values.extend(["failure_zone", "avoid_unless_targeted", "non_colliding_sensor"])
    if semantic == "floor_switch":
        values.extend(["mechanism_trigger", "enterable_zone", "non_colliding_sensor"])
    if semantic == "gate":
        values.extend(["mechanism", "vertical_slider"])
        values.append("passable" if (mechanism_progress or 0.0) >= 0.8 else "blocking_when_closed")
    return list(dict.fromkeys(values))


def objective_focus(
    *,
    trial: dict[str, Any],
    objective_state: dict[str, Any],
    objects: list[SceneObject3D],
) -> dict[str, Any]:
    """Return the current unmet typed objective and its matching scene objects."""

    group = trial.get("objective") or {}
    authored = {
        str(check.get("id")): check
        for check in group.get("checks") or []
        if isinstance(check, dict) and check.get("id")
    }
    results = {
        str(result.get("id")): result
        for result in objective_state.get("checks") or []
        if isinstance(result, dict) and result.get("id")
    }
    ordered = [str(value) for value in group.get("ordered_check_ids") or []]
    ordered_steps = list(objective_state.get("ordered_steps") or [])
    has_ordered_witnesses = len(ordered_steps) == len(ordered)
    ordered_witnesses = (
        {check_id: ordered_steps[index] for index, check_id in enumerate(ordered)}
        if has_ordered_witnesses
        else {}
    )
    ordered.extend(check_id for check_id in authored if check_id not in ordered)
    candidates: list[dict[str, Any]] = []
    for check_id in ordered:
        check = authored.get(check_id) or {}
        result = results.get(check_id) or {}
        phase_passed = bool(result.get("passed"))
        if check_id in ordered_witnesses:
            phase_passed = ordered_witnesses[check_id] is not None
        if phase_passed:
            continue
        predicate = _predicate_for_check(check)
        subject_selector = predicate.get("subject")
        target_selector = predicate.get("target")
        subjects = (
            select_scene_objects(objects, subject_selector)
            if isinstance(subject_selector, dict)
            else []
        )
        targets = (
            select_scene_objects(objects, target_selector)
            if isinstance(target_selector, dict)
            else []
        )
        if predicate.get("type") == "mechanism_state":
            trigger_selector = check.get("selector")
            targets = (
                select_scene_objects(objects, trigger_selector)
                if isinstance(trigger_selector, dict)
                else []
            )
        interaction_kind = _interaction_kind(predicate, subjects, targets)
        navigation_targets = _navigation_targets(
            predicate=predicate,
            subjects=subjects,
            targets=targets,
            interaction_kind=interaction_kind,
        )
        candidates.append(
            {
                "check_id": check_id,
                "type": predicate.get("type") or check.get("type"),
                "description": check.get("description") or "",
                "temporal": check.get("temporal") or "eventually",
                "relation": predicate.get("relation"),
                "subject_ids": [subject.id for subject in subjects],
                "target_ids": [target.id for target in targets],
                "navigation_target_ids": [target.id for target in navigation_targets],
                "subject_semantic_types": sorted({subject.semantic_type for subject in subjects}),
                "target_semantic_types": sorted({target.semantic_type for target in targets}),
                "interaction_kind": interaction_kind,
                "predicate": dict(predicate),
                "capabilities": _capabilities_for_predicate(
                    predicate,
                    interaction_kind=interaction_kind,
                ),
                "current_metrics": result.get("metrics") or {},
                "authoritative_check": {
                    "passed": phase_passed,
                    "status": result.get("status") or "fail",
                    "provisional": bool(result.get("provisional")),
                },
            }
        )
    _enrich_manipulation_destinations(candidates)
    if not candidates:
        return {
            "satisfied": bool(objective_state.get("satisfied")),
            "check_id": None,
            "type": None,
            "subject_ids": [],
            "target_ids": [],
            "navigation_target_ids": [],
            "interaction_kind": None,
            "capabilities": [],
            "predicate": {},
            "authoritative_check": {
                "passed": bool(objective_state.get("satisfied")),
                "status": "pass" if objective_state.get("satisfied") else "fail",
                "provisional": False,
            },
            "alternatives": [],
        }
    current = dict(candidates[0])
    current.update(
        {
            "satisfied": False,
            "objective_mode": group.get("mode") or "all",
            "expected_outcome": trial.get("expected_outcome") or "should_succeed",
            "alternatives": candidates[1:] if group.get("mode") == "any" else [],
        }
    )
    return current


def _enrich_manipulation_destinations(candidates: list[dict[str, Any]]) -> None:
    """Reuse a later destination when an earlier assertion only requires motion."""

    for index, candidate in enumerate(candidates):
        if candidate.get("interaction_kind") != "move_subject":
            continue
        subject_ids = set(candidate.get("subject_ids") or [])
        destination = next(
            (
                later
                for later in candidates[index + 1 :]
                if later.get("interaction_kind") == "move_subject_to_target"
                and subject_ids.intersection(later.get("subject_ids") or [])
                and later.get("target_ids")
            ),
            None,
        )
        if destination is None:
            continue
        candidate["interaction_kind"] = "move_subject_to_target"
        candidate["target_ids"] = list(destination.get("target_ids") or [])
        candidate["target_semantic_types"] = list(
            destination.get("target_semantic_types") or []
        )
        candidate["destination_hint_check_id"] = destination.get("check_id")
        candidate["guidance_predicate"] = dict(destination.get("predicate") or {})
        candidate["capabilities"] = [
            "navigate",
            "stage_behind_object",
            "push",
            "position_object",
        ]


def interaction_guidance(
    *,
    objects: list[SceneObject3D],
    focus: dict[str, Any],
    agent_position: tuple[float, float, float],
    agent_radius: float,
    contact_ids: set[str],
    camera_azimuth: float,
) -> dict[str, Any]:
    """Describe predicate-aware, advisory manipulation geometry."""

    if focus.get("interaction_kind") != "move_subject_to_target":
        return {
            "applicable": False,
            "advisory_only": True,
            "reason": "The current assertion does not require moving another object to a target.",
        }
    by_id = {obj.id: obj for obj in objects}
    subject = next(
        (by_id[value] for value in focus.get("subject_ids") or [] if value in by_id),
        None,
    )
    target = next(
        (by_id[value] for value in focus.get("target_ids") or [] if value in by_id),
        None,
    )
    if subject is None or target is None:
        return {
            "applicable": False,
            "advisory_only": True,
            "reason": "The manipulation assertion does not resolve to one current subject and target.",
        }
    objective_predicate = dict(focus.get("predicate") or {})
    predicate = dict(focus.get("guidance_predicate") or objective_predicate)
    predicate.setdefault("type", focus.get("type"))
    if focus.get("relation") and not predicate.get("relation"):
        predicate["relation"] = focus["relation"]
    placement, destination_kind = _manipulation_placement(
        subject=subject,
        target=target,
        predicate=predicate,
    )
    if placement is None:
        return {
            "applicable": False,
            "advisory_only": True,
            "kind": "move_subject_to_target",
            "subject_id": subject.id,
            "target_id": target.id,
            "reason": (
                "The typed predicate remains fully supported by the evaluator, but it does not "
                "have a generic horizontal manipulation pose."
            ),
        }

    delta = (
        placement.center[0] - subject.position[0],
        placement.center[1] - subject.position[1],
    )
    direction = _normalized_xy(
        delta,
        fallback=(-placement.outward_normal[0], -placement.outward_normal[1]),
    )
    subject_extent = support_radius_xy(subject, direction)
    staging_offset = subject_extent + agent_radius + 0.16
    staging = (
        subject.position[0] - direction[0] * staging_offset,
        subject.position[1] - direction[1] * staging_offset,
    )
    staging_relative = _relative_waypoint(
        origin=agent_position,
        waypoint=staging,
        camera_azimuth=camera_azimuth,
    )
    push_relative = _relative_waypoint(
        origin=subject.position,
        waypoint=placement.center,
        camera_azimuth=camera_azimuth,
    )
    geometric_destination_reached = _geometric_predicate_holds(
        subject=subject,
        target=target,
        predicate=predicate,
    )
    authoritative = dict(focus.get("authoritative_check") or {})
    authoritative_passed = bool(authoritative.get("passed"))
    in_contact = subject.id in contact_ids
    staging_distance = math.dist(agent_position[:2], staging)
    if authoritative_passed:
        phase = "objective_satisfied"
    elif geometric_destination_reached:
        phase = "awaiting_objective_evidence"
    elif in_contact:
        phase = "push_toward_target"
    elif staging_distance <= 0.35:
        phase = "align_and_contact"
    else:
        phase = "approach_staging_point"
    return {
        "applicable": True,
        "advisory_only": True,
        "kind": "move_subject_to_target",
        "phase": phase,
        "subject_id": subject.id,
        "target_id": target.id,
        "predicate_type": predicate.get("type"),
        "objective_predicate_type": objective_predicate.get("type"),
        "relation": predicate.get("relation"),
        "destination_kind": destination_kind,
        "authoritative_check": authoritative,
        "authoritative_check_satisfied": authoritative_passed,
        "geometric_destination_reached": geometric_destination_reached,
        "agent_in_contact_with_subject": in_contact,
        "destination_distance_xy": round(placement.distance, 4),
        "desired_subject_center": [round(placement.center[0], 4), round(placement.center[1], 4)],
        "target_surface_point": [round(placement.target_point[0], 4), round(placement.target_point[1], 4)],
        "staging_position": [round(staging[0], 4), round(staging[1], 4)],
        "staging_relative": staging_relative,
        "staging_distance_xy": round(staging_distance, 4),
        "push_direction_world": [round(direction[0], 5), round(direction[1], 5)],
        "push_direction_relative": push_relative,
        "reason": (
            "Approach the shape-aware staging point, align the subject with the typed "
            "predicate's destination, then push in short batches. Only objective checks "
            "reported by the evaluator establish completion."
        ),
    }


def _manipulation_placement(
    *,
    subject: SceneObject3D,
    target: SceneObject3D,
    predicate: dict[str, Any],
) -> tuple[Placement2D | None, str]:
    predicate_type = str(predicate.get("type") or "")
    relation = str(predicate.get("relation") or "")
    if predicate_type == "overlap" or relation in {"inside", "on_surface", "above", "below"}:
        return nearest_interior_center(subject, target), "interior"
    if predicate_type == "contact" or relation == "near":
        return nearest_contact_center(subject, target), "contact_surface"
    if relation in {"left_of", "right_of", "in_front_of", "behind"}:
        return _axis_relation_placement(subject, target, relation), "directional_relation"
    if relation == "far_from":
        return _far_relation_placement(subject, target, predicate), "distance_relation"
    if predicate_type == "relation":
        return None, "unsupported_advisory_geometry"
    return nearest_contact_center(subject, target), "contact_surface"


def _axis_relation_placement(
    subject: SceneObject3D,
    target: SceneObject3D,
    relation: str,
) -> Placement2D:
    center = [float(subject.position[0]), float(subject.position[1])]
    normal = {
        "left_of": (-1.0, 0.0),
        "right_of": (1.0, 0.0),
        "in_front_of": (0.0, 1.0),
        "behind": (0.0, -1.0),
    }[relation]
    subject_extent = support_radius_xy(subject, normal)
    if relation == "left_of":
        center[0] = target.bounds["x1"] - subject_extent - 0.05
    elif relation == "right_of":
        center[0] = target.bounds["x2"] + subject_extent + 0.05
    elif relation == "in_front_of":
        center[1] = target.bounds["y2"] + subject_extent + 0.05
    else:
        center[1] = target.bounds["y1"] - subject_extent - 0.05
    value = (center[0], center[1])
    return Placement2D(
        center=value,
        target_point=value,
        outward_normal=normal,
        distance=math.dist(subject.position[:2], value),
    )


def _far_relation_placement(
    subject: SceneObject3D,
    target: SceneObject3D,
    predicate: dict[str, Any],
) -> Placement2D:
    delta = (
        float(subject.position[0]) - float(target.position[0]),
        float(subject.position[1]) - float(target.position[1]),
    )
    direction = _normalized_xy(delta, fallback=(1.0, 0.0))
    distance = max(
        float(predicate.get("min_distance") or 0.0) + 0.05,
        math.hypot(*delta),
    )
    center = (
        float(target.position[0]) + direction[0] * distance,
        float(target.position[1]) + direction[1] * distance,
    )
    return Placement2D(
        center=center,
        target_point=(float(target.position[0]), float(target.position[1])),
        outward_normal=direction,
        distance=math.dist(subject.position[:2], center),
    )


def _geometric_predicate_holds(
    *,
    subject: SceneObject3D,
    target: SceneObject3D,
    predicate: dict[str, Any],
) -> bool:
    predicate_type = str(predicate.get("type") or "")
    if predicate_type in {"contact", "overlap"}:
        return volumes_overlap(subject, target)
    if predicate_type == "relation" and predicate.get("relation"):
        return spatial_relation_holds(
            subject,
            target,
            relation=str(predicate["relation"]),
            margin=float(predicate.get("margin") or 0.0),
            min_distance=predicate.get("min_distance"),
            max_distance=predicate.get("max_distance"),
        )
    return False


def _normalized_xy(
    value: tuple[float, float],
    *,
    fallback: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    distance = math.hypot(*value)
    if distance <= 1e-9:
        return fallback
    return value[0] / distance, value[1] / distance


def scene_context(
    *,
    spec: EnvSpec3D,
    trial: dict[str, Any],
    objects: list[SceneObject3D],
    focus: dict[str, Any],
) -> dict[str, Any]:
    """Build compact intent and scene context that is stable across observations."""

    counts = Counter(obj.semantic_type for obj in objects if obj.visible)
    game = spec.game
    mechanisms = [
        {
            "id": mechanism.id,
            "trigger_id": mechanism.trigger_id,
            "gate_id": mechanism.gate_id,
        }
        for mechanism in spec.mechanisms
    ]
    return {
        "environment_request": str(trial.get("environment_request") or "")[:4_000],
        "scene_description": str(trial.get("scene_description") or spec.description)[:1_000],
        "fallback_trial": bool(trial.get("fallback")),
        "object_count": len(objects),
        "semantic_counts": dict(sorted(counts.items())),
        "play_bounds": list(game.play_bounds) if game is not None else _support_dimensions(objects, spec),
        "objective_subject_ids": list(focus.get("subject_ids") or []),
        "objective_target_ids": list(focus.get("target_ids") or []),
        "failure_zone_ids": [obj.id for obj in objects if obj.semantic_type == "hazard"],
        "mechanisms": mechanisms,
    }


def object_clearance_xy(
    point: tuple[float, float, float] | list[float],
    obj: SceneObject3D,
    *,
    radius: float = 0.0,
) -> float:
    """Signed horizontal clearance from a circular actor to an exact footprint."""

    return footprint_signed_clearance(point, obj, radius=radius)


def ground_route_guidance(
    *,
    spec: EnvSpec3D,
    objects: list[SceneObject3D],
    focus: dict[str, Any],
    agent_position: tuple[float, float, float],
    camera_azimuth: float,
    mechanism_states: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Plan an advisory ground route to the current objective's interaction area.

    This is intentionally not an oracle action policy. It only answers whether a
    collision-aware ground approach appears available and supplies sparse
    waypoints. Jumping, pushing, mechanism use, and final objective satisfaction
    remain the child's responsibility.
    """

    target_ids = [
        str(value)
        for value in focus.get("navigation_target_ids") or focus.get("target_ids") or []
    ]
    if not target_ids:
        return _unavailable_route("The current objective has no scene-object target.")
    relation = str(focus.get("relation") or "")
    if relation in {"far_from", "above", "below"}:
        return _unavailable_route(
            f"The {relation} relation does not have a single useful ground-route destination."
        )
    by_id = {obj.id: obj for obj in objects}
    target = next((by_id[target_id] for target_id in target_ids if target_id in by_id), None)
    if target is None:
        return _unavailable_route("No current objective target exists in the scene snapshot.")
    agent = next((obj for obj in objects if obj.semantic_type == "agent"), None)
    if agent is None:
        return _unavailable_route("The scene has no controllable agent.")

    radius = max(agent.size[0], agent.size[1]) * 0.5
    current_agent = scene_object_at(agent, agent_position)
    bounds = _planning_bounds(spec, objects, radius)
    mechanism_progress = _mechanism_progress_by_gate(spec, mechanism_states)
    obstacles = _planning_obstacles(
        objects,
        target=target,
        agent_radius=radius,
        mechanism_progress=mechanism_progress,
    )
    destinations = _objective_destinations(
        focus=focus,
        target=target,
        subject=current_agent,
    )
    if not destinations:
        return _unavailable_route("The current objective has no ground-approach destination.")

    grid = _RouteGrid(bounds=bounds, obstacles=obstacles, resolution=ROUTE_GRID_SIZE)
    if grid.cell_count > ROUTE_MAX_CELLS:
        return _unavailable_route("The scene is too large for compact advisory route planning.")
    authored_start = grid.cell((agent_position[0], agent_position[1]))
    start = grid.nearest_free((agent_position[0], agent_position[1]))
    goals = {
        cell
        for destination in destinations
        if (cell := grid.nearest_free(destination, max_radius=10)) is not None
    }
    if start is None or not goals:
        return {
            **_unavailable_route("No collision-free start or objective approach cell is available."),
            "status": "no_route",
            "target_id": target.id,
        }

    path = grid.find_path(start, goals)
    direct_destination = min(destinations, key=lambda point: math.dist(agent_position[:2], point))
    blocked_by = _segment_blockers(agent_position[:2], direct_destination, obstacles)
    if not path:
        return {
            "available": False,
            "advisory_only": True,
            "mode": "ground_route",
            "status": "no_route",
            "target_id": target.id,
            "target_semantic_type": target.semantic_type,
            "direct_path_blocked": bool(blocked_by),
            "blocked_by": blocked_by,
            "reason": "No ground route was found. The objective may require jumping, moving an object, or changing a mechanism.",
        }

    recovering_clearance = (
        grid.point_blocked((agent_position[0], agent_position[1]))
        or start != authored_start
    )
    points = grid.simplify(path)
    if (
        points
        and not recovering_clearance
        and math.dist(agent_position[:2], points[0]) <= grid.resolution * 1.5
    ):
        points = points[1:]
    if not points:
        points = [grid.point(path[-1])]
    next_waypoint = points[0]
    relative = _relative_waypoint(
        origin=agent_position,
        waypoint=next_waypoint,
        camera_azimuth=camera_azimuth,
    )
    return {
        "available": True,
        "advisory_only": True,
        "mode": "ground_route",
        "status": (
            "clearance_recovery"
            if recovering_clearance
            else "waypoints" if blocked_by else "direct_clear"
        ),
        "target_id": target.id,
        "target_semantic_type": target.semantic_type,
        "direct_path_blocked": bool(blocked_by),
        "blocked_by": blocked_by,
        "next_waypoint": [round(next_waypoint[0], 4), round(next_waypoint[1], 4)],
        "next_waypoint_relative": relative,
        "waypoints": [[round(x, 4), round(y, 4)] for x, y in points[:12]],
        "path_length": round(_path_length([(agent_position[0], agent_position[1]), *points]), 4),
        "reason": (
            "Move to the first waypoint to recover safe clearance before continuing toward the objective."
            if recovering_clearance
            else "Follow sparse waypoints for ground approach, then use objective-specific interaction behavior."
        ),
    }


def action_guidance(
    *,
    agent_position: tuple[float, float, float],
    agent_radius: float,
    objects: list[SceneObject3D],
    focus: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any]:
    """Recommend observation frequency near risk or interaction boundaries."""

    hazards = [obj for obj in objects if obj.semantic_type == "hazard"]
    hazard_clearance = min(
        (object_clearance_xy(agent_position, obj, radius=agent_radius) for obj in hazards),
        default=math.inf,
    )
    target_ids = set(
        focus.get("navigation_target_ids") or focus.get("target_ids") or []
    )
    target_clearance = min(
        (
            object_clearance_xy(agent_position, obj, radius=agent_radius)
            for obj in objects
            if obj.id in target_ids
        ),
        default=math.inf,
    )
    waypoint_distance = float(
        ((route.get("next_waypoint_relative") or {}).get("distance") or math.inf)
    )
    if hazard_clearance <= NEAR_RISK_CLEARANCE:
        frames = 4
        reason = "A failure-zone boundary is close; use fine control and re-observe frequently."
    elif target_clearance <= INTERACTION_RANGE or waypoint_distance <= 0.75:
        frames = 6
        reason = "An interaction target or route turn is close; use a short action batch."
    else:
        frames = 18
        reason = "No immediate risk or interaction boundary is close."
    return {
        "advisory_only": True,
        "recommended_max_frames": frames,
        "reason": reason,
        "nearest_failure_zone_clearance": (
            None if math.isinf(hazard_clearance) else round(hazard_clearance, 4)
        ),
        "current_target_clearance": (
            None if math.isinf(target_clearance) else round(target_clearance, 4)
        ),
    }


def _predicate_for_check(check: dict[str, Any]) -> dict[str, Any]:
    predicate = check.get("predicate")
    if isinstance(predicate, dict):
        return predicate
    check_type = str(check.get("type") or "")
    agent = {"semantic_type": "agent"}
    if check_type == "object_displacement":
        return {"type": "displacement", "subject": check.get("selector") or {}}
    if check_type == "zone_entry":
        return {
            "type": "overlap",
            "subject": agent,
            "target": check.get("selector") or {},
        }
    if check_type == "contact_count":
        return {
            "type": "contact",
            "subject": agent,
            "target": check.get("selector") or {},
        }
    if check_type == "agent_relation":
        return {
            "type": "relation",
            "subject": agent,
            "target": check.get("target") or {},
            "relation": check.get("relation"),
        }
    if check_type == "agent_displacement":
        return {"type": "displacement", "subject": agent}
    if check_type == "agent_height_gain":
        return {"type": "axis_delta", "subject": agent, "axis": "z"}
    if check_type == "jump_count":
        return {"type": "jump_count"}
    if check_type == "mechanism_state":
        return {
            "type": "mechanism_state",
            "mechanism_id": check.get("mechanism_id"),
        }
    if check_type == "terminal_event":
        return {"type": "terminal_event", "event": check.get("event")}
    if check_type == "attempt_reset":
        return {"type": "reset_event", "reason": check.get("reason")}
    return {"type": check_type}


def _interaction_kind(
    predicate: dict[str, Any],
    subjects: list[SceneObject3D],
    targets: list[SceneObject3D],
) -> str | None:
    movable_subjects = [
        subject
        for subject in subjects
        if subject.semantic_type != "agent" and subject.body_type == "dynamic"
    ]
    if not movable_subjects:
        return None
    if predicate.get("type") in {"overlap", "relation", "contact"} and targets:
        return "move_subject_to_target"
    if predicate.get("type") in {"displacement", "axis_delta", "axis_value"}:
        return "move_subject"
    return None


def _navigation_targets(
    *,
    predicate: dict[str, Any],
    subjects: list[SceneObject3D],
    targets: list[SceneObject3D],
    interaction_kind: str | None,
) -> list[SceneObject3D]:
    if interaction_kind in {"move_subject", "move_subject_to_target"}:
        return [subject for subject in subjects if subject.body_type == "dynamic"]
    if predicate.get("type") == "mechanism_state":
        return targets
    if any(subject.semantic_type == "agent" for subject in subjects):
        return targets
    return subjects or targets


def _capabilities_for_predicate(
    predicate: dict[str, Any],
    *,
    interaction_kind: str | None,
) -> list[str]:
    check_type = predicate.get("type")
    relation = predicate.get("relation")
    if interaction_kind == "move_subject_to_target":
        return ["navigate", "stage_behind_object", "push", "position_object"]
    if interaction_kind == "move_subject":
        return ["navigate", "approach_object", "push"]
    if check_type == "displacement":
        return ["locomotion"]
    if check_type == "axis_delta" and predicate.get("axis") == "z":
        return ["jump_or_climb"]
    if check_type == "jump_count":
        return ["jump"]
    if check_type == "overlap":
        return ["navigate", "enter_zone"]
    if check_type == "contact":
        return ["navigate", "make_contact"]
    if check_type == "mechanism_state":
        return ["navigate", "activate_mechanism"]
    if check_type == "terminal_event":
        return ["observe_terminal_outcome"]
    if check_type == "reset_event":
        return ["reset_after_attempt"]
    if check_type == "relation":
        if relation in {"on_surface", "above"}:
            return ["navigate", "approach_surface", "jump_or_climb"]
        if relation == "far_from":
            return ["move_away", "position"]
        return ["navigate", "position"]
    return []

def _support_dimensions(objects: list[SceneObject3D], spec: EnvSpec3D) -> list[float]:
    supports = [obj for obj in objects if obj.semantic_type == "ground"]
    if supports:
        largest = max(supports, key=lambda obj: obj.size[0] * obj.size[1])
        return [largest.size[0], largest.size[1], spec.world_size[2]]
    return list(spec.world_size)


def _planning_bounds(
    spec: EnvSpec3D,
    objects: list[SceneObject3D],
    agent_radius: float,
) -> tuple[float, float, float, float]:
    grounds = [obj for obj in objects if obj.semantic_type == "ground"]
    if grounds:
        ground = max(grounds, key=lambda obj: obj.size[0] * obj.size[1])
        bounds = (ground.bounds["x1"], ground.bounds["x2"], ground.bounds["y1"], ground.bounds["y2"])
    elif spec.game is not None:
        width, depth = spec.game.play_bounds[:2]
        bounds = (-width / 2.0, width / 2.0, -depth / 2.0, depth / 2.0)
    else:
        width, depth = spec.world_size[:2]
        bounds = (-width / 2.0, width / 2.0, -depth / 2.0, depth / 2.0)
    inset = agent_radius + 0.05
    return (bounds[0] + inset, bounds[1] - inset, bounds[2] + inset, bounds[3] - inset)


def _mechanism_progress_by_gate(
    spec: EnvSpec3D,
    states: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, float]:
    if isinstance(states, dict):
        raw_states = list(states.values())
    else:
        raw_states = states
    by_mechanism = {
        str(state.get("id") or state.get("mechanism_id")): float(state.get("progress") or 0.0)
        for state in raw_states
        if isinstance(state, dict)
    }
    return {
        mechanism.gate_id: by_mechanism.get(mechanism.id, 0.0)
        for mechanism in spec.mechanisms
    }


def _planning_obstacles(
    objects: list[SceneObject3D],
    *,
    target: SceneObject3D,
    agent_radius: float,
    mechanism_progress: dict[str, float],
) -> list[_Obstacle]:
    values: list[_Obstacle] = []
    expansion = agent_radius + ROUTE_CLEARANCE_MARGIN
    for obj in objects:
        if obj.semantic_type in {"agent", "ground", "ramp"}:
            continue
        kind = ""
        if obj.semantic_type == "hazard" and obj.id != target.id:
            kind = "failure_zone"
        elif obj.semantic_type == "gate" and mechanism_progress.get(obj.id, 0.0) < 0.8:
            kind = "closed_gate"
        elif obj.semantic_type in STATIC_BLOCKER_TYPES:
            kind = "static_blocker"
        elif obj.body_type == "dynamic" and obj.id != target.id:
            kind = "movable_obstacle"
        elif obj.id == target.id and obj.body_type not in {"sensor"}:
            kind = "objective_object"
        if not kind:
            continue
        values.append(
            _Obstacle(
                object_id=obj.id,
                kind=kind,
                obj=obj,
                clearance=expansion,
            )
        )
    return values


def _objective_destinations(
    *,
    focus: dict[str, Any],
    target: SceneObject3D,
    subject: SceneObject3D,
) -> list[tuple[float, float]]:
    approach_point = focus.get("approach_point")
    if isinstance(approach_point, (list, tuple)) and len(approach_point) >= 2:
        return [(float(approach_point[0]), float(approach_point[1]))]
    relation = str(focus.get("relation") or "")
    if relation in {"left_of", "right_of", "in_front_of", "behind"}:
        return [_axis_relation_placement(subject, target, relation).center]
    if target.body_type == "sensor" or relation in {"inside", "on_surface"}:
        return [nearest_interior_center(subject, target).center]
    return contact_approach_centers(subject, target, clearance=0.35)


class _RouteGrid:
    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        obstacles: list[_Obstacle],
        resolution: float,
    ) -> None:
        self.bounds = bounds
        self.obstacles = obstacles
        self.resolution = resolution
        self.width = max(1, int(math.floor((bounds[1] - bounds[0]) / resolution)) + 1)
        self.height = max(1, int(math.floor((bounds[3] - bounds[2]) / resolution)) + 1)
        self.cell_count = self.width * self.height

    def point(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.bounds[0] + cell[0] * self.resolution,
            self.bounds[2] + cell[1] * self.resolution,
        )

    def cell(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            min(self.width - 1, max(0, int(round((point[0] - self.bounds[0]) / self.resolution)))),
            min(self.height - 1, max(0, int(round((point[1] - self.bounds[2]) / self.resolution)))),
        )

    def blocked(self, cell: tuple[int, int]) -> bool:
        return self.point_blocked(self.point(cell))

    def point_blocked(self, point: tuple[float, float]) -> bool:
        return any(
            footprint_signed_clearance(
                point,
                obstacle.obj,
                radius=obstacle.clearance,
            ) <= 0.0
            for obstacle in self.obstacles
        )

    def nearest_free(
        self,
        point: tuple[float, float],
        *,
        max_radius: int = 6,
    ) -> tuple[int, int] | None:
        origin = self.cell(point)
        if not self.blocked(origin):
            return origin
        for radius in range(1, max_radius + 1):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    candidates.append((origin[0] + dx, origin[1] + dy))
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    candidates.append((origin[0] + dx, origin[1] + dy))
            valid = [cell for cell in candidates if self._inside(cell) and not self.blocked(cell)]
            if valid:
                return min(valid, key=lambda cell: math.dist(self.point(cell), point))
        return None

    def find_path(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        costs = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        reached: tuple[int, int] | None = None
        while frontier:
            _estimate, cost, current = heapq.heappop(frontier)
            if cost > costs.get(current, math.inf):
                continue
            if current in goals:
                reached = current
                break
            for neighbor, step_cost in self._neighbors(current):
                candidate_cost = cost + step_cost
                if candidate_cost + 1e-9 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = candidate_cost
                previous[neighbor] = current
                heuristic = min(math.dist(neighbor, goal) for goal in goals) * self.resolution
                heapq.heappush(frontier, (candidate_cost + heuristic, candidate_cost, neighbor))
        if reached is None:
            return []
        path = [reached]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def simplify(self, path: list[tuple[int, int]]) -> list[tuple[float, float]]:
        if not path:
            return []
        selected = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1 and not self._line_clear(path[anchor], path[candidate]):
                candidate -= 1
            selected.append(path[candidate])
            anchor = candidate
        return [self.point(cell) for cell in selected]

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _neighbors(self, cell: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float]]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not self._inside(neighbor) or self.blocked(neighbor):
                continue
            if dx and dy:
                if self.blocked((cell[0] + dx, cell[1])) or self.blocked((cell[0], cell[1] + dy)):
                    continue
            yield neighbor, self.resolution * (math.sqrt(2.0) if dx and dy else 1.0)

    def _line_clear(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        start_point = self.point(start)
        end_point = self.point(end)
        return not any(
            segment_intersects_footprint(
                start_point,
                end_point,
                obstacle.obj,
                clearance=obstacle.clearance,
            )
            for obstacle in self.obstacles
        )


def _relative_waypoint(
    *,
    origin: tuple[float, float, float],
    waypoint: tuple[float, float],
    camera_azimuth: float,
) -> dict[str, float]:
    dx = waypoint[0] - origin[0]
    dy = waypoint[1] - origin[1]
    forward = camera_relative_velocity(camera_azimuth, right=0.0, forward=1.0, speed=1.0)
    right = camera_relative_velocity(camera_azimuth, right=1.0, forward=0.0, speed=1.0)
    desired = math.degrees(math.atan2(-dx, -dy)) if abs(dx) + abs(dy) > 1e-9 else camera_azimuth
    return {
        "forward": round(dx * forward[0] + dy * forward[1], 4),
        "right": round(dx * right[0] + dy * right[1], 4),
        "distance": round(math.hypot(dx, dy), 4),
        "desired_camera_azimuth": round(desired, 4),
        "heading_error_degrees": round(_wrapped_degrees(desired - camera_azimuth), 4),
    }


def _segment_blockers(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[_Obstacle],
) -> list[str]:
    return list(
        dict.fromkeys(
            obstacle.object_id
            for obstacle in obstacles
            if segment_intersects_footprint(
                start,
                end,
                obstacle.obj,
                clearance=obstacle.clearance,
            )
        )
    )


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _unavailable_route(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "advisory_only": True,
        "mode": "ground_route",
        "status": "not_applicable",
        "reason": reason,
    }


def _wrapped_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0
