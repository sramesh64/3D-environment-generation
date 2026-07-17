"""Select concise, general-purpose camera evidence for behavior rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .env_tasks import describe_assertion_condition


SUPPORT_SEMANTIC_TYPES = {"ground", "platform", "ramp"}
EVENT_PRIORITIES = {
    "terminal_event": 120,
    "attempt_reset": 115,
    "mechanism_activated": 100,
    "zone_entered": 95,
    "jump": 80,
    "contact_started": 65,
}


@dataclass(frozen=True)
class EvidenceSelection:
    path: Path
    record: dict[str, Any]
    kind: str
    label: str
    priority: int
    focus_ids: tuple[str, ...] = ()


def select_behavior_evidence(
    paths: Iterable[Path],
    records_by_name: dict[str, dict[str, Any]],
    *,
    limit: int = 6,
) -> list[EvidenceSelection]:
    """Select chronologically ordered frames that explain a rollout.

    Typed objective transitions are preferred over simulator events. Routine
    support contacts are ignored, while closest-approach and evenly spaced
    frames provide useful evidence for inconclusive and event-free trials.
    """

    ordered_paths = list(dict.fromkeys(paths))
    if not ordered_paths or limit <= 0:
        return []
    index_by_path = {path: index for index, path in enumerate(ordered_paths)}
    records = {
        path: records_by_name.get(path.name) or {}
        for path in ordered_paths
    }
    selected: dict[Path, EvidenceSelection] = {}

    def choose(candidate: EvidenceSelection) -> None:
        current = selected.get(candidate.path)
        if current is not None:
            if candidate.priority > current.priority:
                selected[candidate.path] = candidate
            return
        if len(selected) < limit:
            selected[candidate.path] = candidate

    first = ordered_paths[0]
    last = ordered_paths[-1]
    choose(_candidate(first, records[first], "initial", "Initial observation", 20))
    choose(_candidate(last, records[last], "final", _final_label(records[last]), 120))

    transitions = _objective_transition_candidates(ordered_paths, records)
    available_transitions = [item for item in transitions if item.path not in selected]
    for candidate in _spread(available_transitions, max(0, limit - len(selected))):
        choose(candidate)

    for candidate in _event_candidates(ordered_paths, records):
        choose(candidate)
        if len(selected) >= limit:
            break

    for candidate in _attempt_start_candidates(ordered_paths, records):
        choose(candidate)
        if len(selected) >= limit:
            break

    for candidate in _closest_approach_candidates(ordered_paths, records):
        choose(candidate)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for path in _progress_fill_order(ordered_paths, records, selected, limit):
            choose(
                _candidate(
                    path,
                    records[path],
                    "progress",
                    _progress_label(records[path]),
                    10,
                )
            )
            if len(selected) >= limit:
                break

    return sorted(selected.values(), key=lambda item: index_by_path[item.path])


def significant_frame_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-noisy events attached to one frame manifest record."""

    raw_events = record.get("events")
    if not isinstance(raw_events, list):
        raw_events = record.get("recent_events") or []
    return [
        event
        for event in raw_events
        if isinstance(event, dict) and _event_is_significant(event)
    ]


def _candidate(
    path: Path,
    record: dict[str, Any],
    kind: str,
    label: str,
    priority: int,
    focus_ids: Iterable[str] | None = None,
) -> EvidenceSelection:
    resolved_focus_ids = _unique_ids(
        _record_focus_ids(record) if focus_ids is None else focus_ids
    )
    return EvidenceSelection(
        path=path,
        record=record,
        kind=kind,
        label=label,
        priority=priority,
        focus_ids=resolved_focus_ids,
    )


def _objective_transition_candidates(
    paths: list[Path],
    records: dict[Path, dict[str, Any]],
) -> list[EvidenceSelection]:
    previous_objectives: dict[tuple[int, str], bool] = {}
    previous_constraints: dict[tuple[int, str], bool] = {}
    focus_ids_by_check: dict[tuple[int, str], tuple[str, ...]] = {}
    values: list[EvidenceSelection] = []
    for path in paths:
        record = records[path]
        attempt = _int(record.get("attempt"), 1)
        focus = record.get("objective_focus") or {}
        focus_check_id = str(focus.get("check_id") or "")
        if focus_check_id:
            focus_ids_by_check[(attempt, focus_check_id)] = _unique_ids(
                [*(focus.get("subject_ids") or []), *(focus.get("target_ids") or [])]
            )
        objective = record.get("attempt_objective") or record.get("objective") or {}
        for check in objective.get("checks") or []:
            if not isinstance(check, dict) or not check.get("id"):
                continue
            key = (attempt, str(check["id"]))
            passed = bool(check.get("passed"))
            if passed and previous_objectives.get(key) is not True:
                description = _check_name(check)
                values.append(
                    _candidate(
                        path,
                        record,
                        "objective",
                        f"Completed: {description}",
                        110,
                        focus_ids=focus_ids_by_check.get(key) or _record_focus_ids(record),
                    )
                )
            previous_objectives[key] = passed

        constraints = record.get("attempt_constraints") or record.get("constraints") or {}
        for check in constraints.get("checks") or []:
            if not isinstance(check, dict) or not check.get("id"):
                continue
            key = (attempt, str(check["id"]))
            passed = bool(check.get("passed"))
            if previous_constraints.get(key) is True and not passed:
                description = _check_name(check)
                values.append(
                    _candidate(path, record, "constraint", f"Constraint violated: {description}", 115)
                )
            previous_constraints[key] = passed
    return _dedupe_candidates(values)


def _event_candidates(
    paths: list[Path],
    records: dict[Path, dict[str, Any]],
) -> list[EvidenceSelection]:
    seen: set[tuple[int, str, str]] = set()
    values: list[EvidenceSelection] = []
    for path in paths:
        record = records[path]
        attempt = _int(record.get("attempt"), 1)
        significant_events = significant_frame_events(record)
        event_focus_ids = _unique_ids(
            event.get("object_id")
            for event in significant_events
            if event.get("object_id")
        )
        for event in significant_events:
            event_type = str(event.get("type") or "event")
            object_id = str(event.get("object_id") or "")
            key = (attempt, event_type, object_id)
            if key in seen:
                continue
            seen.add(key)
            priority = EVENT_PRIORITIES.get(event_type, 70)
            values.append(
                _candidate(
                    path,
                    record,
                    "event",
                    _event_label(event),
                    priority,
                    focus_ids=event_focus_ids,
                )
            )
    return sorted(
        _dedupe_candidates(values),
        key=lambda item: (-item.priority, _int(item.record.get("total_step"), 0)),
    )


def _attempt_start_candidates(
    paths: list[Path],
    records: dict[Path, dict[str, Any]],
) -> list[EvidenceSelection]:
    seen: set[int] = set()
    values = []
    for path in paths:
        attempt = _int(records[path].get("attempt"), 1)
        if attempt in seen:
            continue
        seen.add(attempt)
        values.append(
            _candidate(path, records[path], "attempt", f"Attempt {attempt} started", 70)
        )
    return values


def _closest_approach_candidates(
    paths: list[Path],
    records: dict[Path, dict[str, Any]],
) -> list[EvidenceSelection]:
    closest: dict[tuple[int, str, str], tuple[float, Path]] = {}
    for path in paths:
        record = records[path]
        objective = record.get("attempt_objective") or record.get("objective") or {}
        if objective.get("satisfied"):
            continue
        navigation = record.get("navigation") or {}
        if not navigation.get("available"):
            continue
        distance = _float(navigation.get("distance_xy"))
        if distance is None:
            continue
        focus = record.get("objective_focus") or {}
        focus_id = str(focus.get("check_id") or navigation.get("primary_target_id") or "target")
        target_id = str(navigation.get("primary_target_id") or focus_id)
        key = (_int(record.get("attempt"), 1), focus_id, target_id)
        current = closest.get(key)
        if current is None or distance < current[0]:
            closest[key] = (distance, path)
    values = [
        _candidate(
            path,
            records[path],
            "approach",
            f"Closest approach: {_humanize(target_id)}",
            60,
        )
        for (_attempt, _focus_id, target_id), (_distance, path) in closest.items()
    ]
    return sorted(values, key=lambda item: _int(item.record.get("total_step"), 0))


def _progress_fill_order(
    paths: list[Path],
    records: dict[Path, dict[str, Any]],
    selected: dict[Path, EvidenceSelection],
    limit: int,
) -> list[Path]:
    if not paths:
        return []
    available = {index for index, path in enumerate(paths) if path not in selected}
    selected_indices = {index for index, path in enumerate(paths) if path in selected}
    values: list[Path] = []
    while available and len(selected) + len(values) < limit:
        clean = {
            index
            for index in available
            if not _has_only_noisy_events(records[paths[index]])
        }
        candidates = clean or available
        next_index = max(
            candidates,
            key=lambda index: (
                min((abs(index - chosen) for chosen in selected_indices), default=len(paths)),
                -index,
            ),
        )
        available.remove(next_index)
        selected_indices.add(next_index)
        values.append(paths[next_index])
    values.extend(
        paths[index]
        for index in sorted(
            available,
            key=lambda index: (_has_only_noisy_events(records[paths[index]]), index),
        )
    )
    return values


def _spread(values: list[EvidenceSelection], count: int) -> list[EvidenceSelection]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indices)]


def _dedupe_candidates(values: list[EvidenceSelection]) -> list[EvidenceSelection]:
    by_path: dict[Path, EvidenceSelection] = {}
    order: list[Path] = []
    for value in values:
        current = by_path.get(value.path)
        if current is None:
            order.append(value.path)
            by_path[value.path] = value
        elif value.priority > current.priority:
            by_path[value.path] = value
    return [by_path[path] for path in order]


def _event_is_significant(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    if event_type != "contact_started":
        return bool(event_type)
    if event.get("routine") is True:
        return False
    if event.get("objective_relevant") is True:
        return True
    semantic_type = str(event.get("semantic_type") or "")
    return bool(semantic_type) and semantic_type not in SUPPORT_SEMANTIC_TYPES


def _record_focus_ids(record: dict[str, Any]) -> tuple[str, ...]:
    focus = record.get("objective_focus") or {}
    values: list[Any] = [
        *(focus.get("subject_ids") or []),
        *(focus.get("target_ids") or []),
    ]
    interaction = record.get("interaction_guidance") or {}
    values.extend([interaction.get("subject_id"), interaction.get("target_id")])
    navigation_id = (record.get("navigation") or {}).get("primary_target_id")
    if navigation_id:
        values.append(navigation_id)
    values.extend(
        event.get("object_id")
        for event in significant_frame_events(record)
        if event.get("object_id")
    )
    return _unique_ids(values)


def _unique_ids(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value)
            for value in values
            if value is not None and str(value).strip()
        )
    )


def _has_only_noisy_events(record: dict[str, Any]) -> bool:
    raw = record.get("events")
    if not isinstance(raw, list):
        raw = record.get("recent_events") or []
    return bool(raw) and not significant_frame_events(record)


def _event_label(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "event")
    object_name = _humanize(str(event.get("object_id") or ""))
    if event_type == "mechanism_activated":
        return f"Activated {object_name}" if object_name else "Mechanism activated"
    if event_type == "zone_entered":
        return f"Entered {object_name}" if object_name else "Entered zone"
    if event_type == "terminal_event":
        outcome = _humanize(str(event.get("outcome") or "terminal"))
        return f"Attempt ended: {outcome}"
    if event_type == "attempt_reset":
        reason = _humanize(str(event.get("reason") or event.get("outcome") or "manual"))
        return f"Reset after {reason}"
    if event_type == "contact_started":
        return f"Contacted {object_name}" if object_name else "Made contact"
    if event_type == "jump":
        return "Jumped"
    label = _humanize(event_type)
    return f"{label}: {object_name}" if object_name else label


def _final_label(record: dict[str, Any]) -> str:
    objective = record.get("attempt_objective") or record.get("objective") or {}
    if objective.get("satisfied"):
        passed = [check for check in objective.get("checks") or [] if check.get("passed")]
        if passed:
            return f"Completed: {_check_name(passed[-1])}"
        return "Objective completed"
    events = significant_frame_events(record)
    if events:
        return _event_label(events[-1])
    reason = str(record.get("termination_reason") or "").strip()
    return f"Ended: {_humanize(reason)}" if reason else "Final observation"


def _progress_label(record: dict[str, Any]) -> str:
    focus = record.get("objective_focus") or {}
    description = str(focus.get("description") or "").strip()
    if description:
        return f"Progress: {description}"
    check_id = str(focus.get("check_id") or "").strip()
    return f"Progress: {_humanize(check_id)}" if check_id else "Rollout progress"


def _check_name(check: dict[str, Any]) -> str:
    return describe_assertion_condition(check).rstrip(". ")


def _humanize(value: str) -> str:
    return " ".join(value.replace("-", "_").split("_")).strip()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
