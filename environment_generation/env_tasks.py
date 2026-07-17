"""Persistent user tasks and their typed trajectory-test definitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env_verification import (
    SUPPORTED_SPATIAL_RELATIONS,
    normalize_selector,
    scene_objects,
    select_scene_objects,
)


TASKS_DIRNAME = "tasks"
ENV_SPEC_FILENAME = "env_spec_3d.json"
TASK_SCHEMA_VERSION = "1.0"
TASK_CONTROLLER_VERSION = "task_controller_v1"
TASK_FILENAME = "task.json"
TASK_COMPILER_OUTPUT_SCHEMA_PATH = Path(__file__).with_name("task_compiler_output_schema.json")
MAX_TASKS = 32
MAX_TESTS = 16
MAX_CONDITIONS_PER_TEST = 16
MAX_TASK_STEPS = 20_000
MIN_TASK_STEPS = 60
MAX_SUSTAINED_FRAMES = 10_000
MAX_TASK_INSTRUCTION_CHARS = 4_000

TASK_STATUSES = {
    "compiling",
    "pending_oracle",
    "recording",
    "validation_failed",
    "validated",
    "stale",
    "error",
}
SUPPORTED_TEST_MODES = {"all", "any"}
SUPPORTED_TEMPORAL_OPERATORS = {
    "eventually",
    "at_end",
    "always",
    "never",
    "sustained",
    "count",
}
SUPPORTED_PREDICATE_TYPES = {
    "overlap",
    "relation",
    "displacement",
    "axis_delta",
    "contact",
    "axis_value",
    "speed",
    "settled",
    "mechanism_state",
    "jump_count",
    "step_count",
    "reset_count",
    "reset_event",
    "terminal_event",
    "in_bounds",
    "grounded",
}
SUPPORTED_RELATIONS = SUPPORTED_SPATIAL_RELATIONS - {"between"}
SUPPORTED_TERMINAL_EVENTS = {"goal", "hazard", "out_of_bounds"}
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class TaskDefinitionError(ValueError):
    """Raised when a task or trajectory-test definition is malformed."""


_GENERIC_ASSERTION_WORDS = {
    "agent",
    "axis",
    "bounds",
    "check",
    "condition",
    "contact",
    "count",
    "delta",
    "displacement",
    "entry",
    "event",
    "gain",
    "grounded",
    "height",
    "in",
    "jump",
    "mechanism",
    "objective",
    "object",
    "overlap",
    "predicate",
    "relation",
    "reset",
    "settled",
    "speed",
    "state",
    "step",
    "terminal",
    "test",
    "value",
    "zone",
}


def describe_assertion_condition(
    condition: dict[str, Any],
    *,
    fallback: str | None = None,
) -> str:
    """Return a concise user-facing label for a typed trajectory condition."""

    predicate = condition.get("predicate") if isinstance(condition.get("predicate"), dict) else {}
    predicate_type = str(
        predicate.get("type")
        or condition.get("predicate_type")
        or condition.get("type")
        or ""
    ).strip().lower()
    for candidate in (condition.get("description"), fallback):
        text = " ".join(str(candidate or "").split())
        if text and not _generic_assertion_description(text, predicate_type):
            return text[:600]

    subject = _selector_description(predicate.get("subject"), default="Robot")
    target = _selector_description(predicate.get("target"), default="target", capitalize=False)
    phrase = _predicate_description(predicate_type, predicate, subject=subject, target=target)
    if not phrase:
        phrase = _humanize_identifier(condition.get("id") or predicate_type or "objective")

    temporal = str(condition.get("temporal") or "eventually").strip().lower()
    suffix = {
        "always": " throughout the attempt",
        "never": " at no point during the attempt",
        "at_end": " at the end of the attempt",
        "sustained": " for the required duration",
        "count": " the required number of times",
    }.get(temporal, "")
    return f"{phrase}{suffix}."[:600]


def _generic_assertion_description(value: str, predicate_type: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if not normalized:
        return True
    if normalized == predicate_type.replace("_", " "):
        return True
    return set(normalized.split()) <= _GENERIC_ASSERTION_WORDS


def _selector_description(value: Any, *, default: str, capitalize: bool = True) -> str:
    if isinstance(value, str):
        return _humanize_identifier(value, capitalize=capitalize)
    if not isinstance(value, dict):
        return default
    for key in ("id", "semantic_type", "object_type", "body_type", "shape", "tag"):
        if value.get(key):
            return _humanize_identifier(value[key], capitalize=capitalize)
    return default


def _humanize_identifier(value: Any, *, capitalize: bool = True) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "")).strip()
    text = " ".join(text.split()) or "objective"
    return (text[0].upper() if capitalize else text[0].lower()) + text[1:]


def _predicate_description(
    predicate_type: str,
    predicate: dict[str, Any],
    *,
    subject: str,
    target: str,
) -> str:
    if predicate_type == "contact":
        return f"{subject} makes contact with {target}"
    if predicate_type == "overlap":
        return f"{subject} enters {target}"
    if predicate_type == "relation":
        relation = str(predicate.get("relation") or "").strip().lower()
        relation_phrase = {
            "above": "is above",
            "below": "is below",
            "behind": "is behind",
            "far_from": "is far from",
            "in_front_of": "is in front of",
            "inside": "is inside",
            "left_of": "is left of",
            "near": "is near",
            "on_surface": "is on top of",
            "right_of": "is right of",
        }.get(relation, f"satisfies the {_humanize_identifier(relation).lower()} relation with")
        return f"{subject} {relation_phrase} {target}"
    if predicate_type == "displacement":
        return f"{subject} moves {_bounds_description(predicate, unit='m')}"
    if predicate_type == "axis_delta":
        axis = str(predicate.get("axis") or "z").lower()
        axis_name = {"x": "x", "y": "y", "z": "vertical"}.get(axis, axis)
        return f"{subject} changes its {axis_name} position {_bounds_description(predicate, unit='m')}"
    if predicate_type == "axis_value":
        axis = str(predicate.get("axis") or "").lower()
        return f"{subject} reaches the required {axis}-axis position"
    if predicate_type == "speed":
        component = str(predicate.get("component") or "linear").lower()
        return f"{subject} has the required {component} speed"
    if predicate_type == "settled":
        return f"{subject} comes to rest"
    if predicate_type == "mechanism_state":
        mechanism = _humanize_identifier(predicate.get("mechanism_id") or "mechanism")
        state = str(predicate.get("state") or "open").replace("_", " ")
        return f"{mechanism} becomes {state}"
    if predicate_type == "jump_count":
        return "The robot jumps"
    if predicate_type == "step_count":
        return "The run advances through the required simulation steps"
    if predicate_type == "reset_count":
        return "The run has the required number of resets"
    if predicate_type == "reset_event":
        reason = str(predicate.get("reason") or "any").replace("_", " ")
        return "A reset occurs" if reason == "any" else f"A {reason} reset occurs"
    if predicate_type == "terminal_event":
        event = str(predicate.get("event") or "terminal").replace("_", " ")
        return f"The run records the {event} event"
    if predicate_type == "in_bounds":
        return f"{subject} remains inside the play bounds"
    if predicate_type == "grounded":
        return f"{subject} is supported by a walkable surface"
    return ""


def _bounds_description(predicate: dict[str, Any], *, unit: str) -> str:
    minimum = predicate.get("min_value")
    maximum = predicate.get("max_value")
    if minimum is not None and maximum is not None:
        return f"between {minimum:g} and {maximum:g} {unit}"
    if minimum is not None:
        return f"at least {minimum:g} {unit}"
    if maximum is not None:
        return f"at most {maximum:g} {unit}"
    return "by the required amount"


def normalize_assertion_group(
    *,
    raw: Any,
    spec: dict[str, Any],
    required: bool = True,
    field_name: str = "assertion group",
) -> dict[str, Any]:
    """Normalize one reusable group of typed trajectory conditions.

    Tasks wrap these groups in named tests. Behavioral trials use the same
    condition contract for objectives and attempt constraints.
    """

    if raw is None and not required:
        return {"mode": "all", "checks": [], "ordered_check_ids": []}
    if not isinstance(raw, dict):
        raise TaskDefinitionError(f"{field_name} must be an object")
    conditions = raw.get("conditions")
    if conditions is None:
        conditions = raw.get("checks")
    if not isinstance(conditions, list) or (required and not conditions):
        requirement = "a non-empty list" if required else "a list"
        raise TaskDefinitionError(f"{field_name} conditions must be {requirement}")
    if not conditions:
        return {"mode": "all", "checks": [], "ordered_check_ids": []}
    objects = scene_objects(spec)
    normalized = _normalize_test(
        0,
        {
            "id": str(raw.get("id") or field_name).replace(" ", "_"),
            "description": str(raw.get("description") or field_name),
            "mode": raw.get("mode") or "all",
            "conditions": conditions,
            "ordered_condition_ids": (
                raw.get("ordered_condition_ids")
                or raw.get("ordered_check_ids")
                or raw.get("ordered")
                or []
            ),
        },
        objects=objects,
        spec=spec,
    )
    return {
        "mode": normalized["mode"],
        "checks": normalized["conditions"],
        "ordered_check_ids": normalized["ordered_condition_ids"],
    }


def normalize_task_definition(
    *,
    env_id: str,
    instruction: str,
    tests: Any,
    spec: dict[str, Any],
    max_steps: Any = 6_000,
) -> dict[str, Any]:
    instruction_text = str(instruction or "").strip()
    if not instruction_text:
        raise TaskDefinitionError("task instruction is required")
    if len(instruction_text) > MAX_TASK_INSTRUCTION_CHARS:
        raise TaskDefinitionError(
            f"task instruction exceeds {MAX_TASK_INSTRUCTION_CHARS} characters"
        )
    if not isinstance(tests, list) or not tests:
        raise TaskDefinitionError("tests must be a non-empty list")
    if len(tests) > MAX_TESTS:
        raise TaskDefinitionError(f"tasks support at most {MAX_TESTS} tests")
    objects = scene_objects(spec)
    normalized_tests = [
        _normalize_test(index, raw, objects=objects, spec=spec)
        for index, raw in enumerate(tests)
    ]
    ids = [test["id"] for test in normalized_tests]
    if len(ids) != len(set(ids)):
        raise TaskDefinitionError("test ids must be unique")
    if not any(
        _condition_is_positive(condition)
        for test in normalized_tests
        for condition in test["conditions"]
    ):
        raise TaskDefinitionError(
            "task needs at least one positive completion condition; invariants alone can be satisfied by doing nothing"
        )
    normalized_tests.extend(
        _system_safety_tests(
            objects=objects,
            spec=spec,
            existing=normalized_tests,
        )
    )
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "env_id": str(env_id),
        "instruction": instruction_text,
        "max_steps": _bounded_int(max_steps, "max_steps", MIN_TASK_STEPS, MAX_TASK_STEPS),
        "tests": normalized_tests,
    }


def create_task_draft(
    *,
    scene_dir: Path,
    env_id: str,
    instruction: str,
    compiler_output: dict[str, Any],
    model: str = "",
    requested_task_id: str = "",
) -> dict[str, Any]:
    spec = _load_final_spec(scene_dir)
    _require_controllable_agent(spec)
    existing = list_tasks(scene_dir, current_spec=spec)
    if len(existing) >= MAX_TASKS:
        raise TaskDefinitionError(f"an environment supports at most {MAX_TASKS} tasks")
    definition = normalize_task_definition(
        env_id=env_id,
        instruction=instruction,
        tests=compiler_output.get("tests"),
        max_steps=compiler_output.get("max_steps", 6_000),
        spec=spec,
    )
    task_id = _unique_task_id(
        scene_dir,
        requested_task_id or str(compiler_output.get("task_id") or instruction),
    )
    now = _now()
    task = {
        **definition,
        "task_id": task_id,
        "status": "pending_oracle",
        "summary": str(compiler_output.get("summary") or instruction).strip()[:1_200],
        "env_spec_hash": task_scene_hash(spec),
        "controller_version": TASK_CONTROLLER_VERSION,
        "task_definition_hash": task_definition_hash(definition),
        "created_at": now,
        "updated_at": now,
        "compiler": {
            "kind": "codex_typed_task_compiler",
            "model": str(model or "default"),
        },
        "oracle": None,
        "last_validation": None,
        "run_summaries": [],
    }
    write_task(scene_dir, task)
    return task


def create_compiling_task(
    *,
    scene_dir: Path,
    env_id: str,
    instruction: str,
    requested_task_id: str = "",
) -> dict[str, Any]:
    spec = _load_final_spec(scene_dir)
    _require_controllable_agent(spec)
    task_id = _unique_task_id(scene_dir, requested_task_id or instruction)
    now = _now()
    task = {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "env_id": env_id,
        "instruction": str(instruction).strip(),
        "status": "compiling",
        "env_spec_hash": task_scene_hash(spec),
        "controller_version": TASK_CONTROLLER_VERSION,
        "created_at": now,
        "updated_at": now,
        "tests": [],
        "max_steps": 6_000,
        "oracle": None,
        "last_validation": None,
        "run_summaries": [],
    }
    write_task(scene_dir, task)
    return task


def finish_compiling_task(
    *,
    scene_dir: Path,
    task_id: str,
    compiler_output: dict[str, Any],
    model: str = "",
    compile_attempts: int = 1,
    repaired_validation_errors: tuple[str, ...] = (),
) -> dict[str, Any]:
    current = read_task(scene_dir, task_id, include_staleness=False)
    spec = _load_final_spec(scene_dir)
    definition = normalize_task_definition(
        env_id=str(current["env_id"]),
        instruction=str(current["instruction"]),
        tests=compiler_output.get("tests"),
        max_steps=compiler_output.get("max_steps", 6_000),
        spec=spec,
    )
    current.update(definition)
    current.update(
        {
            "status": "pending_oracle",
            "summary": str(compiler_output.get("summary") or current["instruction"]).strip()[:1_200],
            "env_spec_hash": task_scene_hash(spec),
            "controller_version": TASK_CONTROLLER_VERSION,
            "task_definition_hash": task_definition_hash(definition),
            "compiler": _task_compiler_metadata(
                model=model,
                attempts=compile_attempts,
                validation_errors=repaired_validation_errors,
            ),
            "compiler_error": None,
        }
    )
    write_task(scene_dir, current)
    return current


def mark_task_compile_error(
    scene_dir: Path,
    task_id: str,
    error: str,
    *,
    model: str = "",
    compile_attempts: int = 1,
    validation_errors: tuple[str, ...] = (),
) -> dict[str, Any]:
    task = read_task(scene_dir, task_id, include_staleness=False)
    task.update(
        {
            "status": "error",
            "compiler_error": str(error)[:2_000],
            "compiler": _task_compiler_metadata(
                model=model,
                attempts=compile_attempts,
                validation_errors=validation_errors,
            ),
        }
    )
    write_task(scene_dir, task)
    return task


def _task_compiler_metadata(
    *,
    model: str,
    attempts: int,
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    bounded_attempts = max(1, int(attempts))
    return {
        "kind": "codex_typed_task_compiler",
        "model": str(model or "default"),
        "attempts": bounded_attempts,
        "repair_attempts": max(0, bounded_attempts - 1),
        "validation_errors": [str(error)[:2_000] for error in validation_errors],
    }


def tasks_dir(scene_dir: Path) -> Path:
    return scene_dir / TASKS_DIRNAME


def task_dir(scene_dir: Path, task_id: str) -> Path:
    _validate_task_id(task_id)
    return tasks_dir(scene_dir) / task_id


def task_path(scene_dir: Path, task_id: str) -> Path:
    return task_dir(scene_dir, task_id) / TASK_FILENAME


def read_task(
    scene_dir: Path,
    task_id: str,
    *,
    current_spec: dict[str, Any] | None = None,
    include_staleness: bool = True,
) -> dict[str, Any]:
    path = task_path(scene_dir, task_id)
    if not path.is_file():
        raise FileNotFoundError(f"task does not exist: {task_id}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskDefinitionError(f"task file is malformed: {task_id}") from exc
    if not isinstance(value, dict):
        raise TaskDefinitionError(f"task file is malformed: {task_id}")
    if include_staleness:
        value = with_task_staleness(value, current_spec or _read_spec(scene_dir))
    return value


def write_task(scene_dir: Path, task: dict[str, Any]) -> Path:
    task_id = str(task.get("task_id") or "")
    _validate_task_id(task_id)
    status = str(task.get("status") or "")
    if status not in TASK_STATUSES:
        raise TaskDefinitionError(f"unsupported task status {status!r}")
    task["updated_at"] = _now()
    path = task_path(scene_dir, task_id)
    _atomic_write_json(path, task)
    return path


def list_tasks(
    scene_dir: Path,
    *,
    current_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    root = tasks_dir(scene_dir)
    if not root.is_dir():
        return []
    spec = current_spec if current_spec is not None else _read_spec(scene_dir)
    values: list[dict[str, Any]] = []
    for path in root.glob(f"*/{TASK_FILENAME}"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not value.get("task_id"):
            continue
        values.append(with_task_staleness(value, spec))
    values.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return values


def task_catalog_summary(scene_dir: Path, *, current_spec: dict[str, Any] | None) -> dict[str, Any]:
    tasks = list_tasks(scene_dir, current_spec=current_spec)
    counts = {status: 0 for status in TASK_STATUSES}
    for task in tasks:
        status = str(task.get("effective_status") or task.get("status") or "error")
        counts[status] = counts.get(status, 0) + 1
    validated = counts.get("validated", 0)
    stale = counts.get("stale", 0)
    pending = len(tasks) - validated - stale
    return {
        "total": len(tasks),
        "validated": validated,
        "stale": stale,
        "pending": pending,
        "counts": counts,
        "label": (
            "No tasks"
            if not tasks
            else f"{validated}/{len(tasks)} validated"
        ),
    }


def with_task_staleness(task: dict[str, Any], current_spec: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(task)
    stale_reason = ""
    if current_spec is None:
        stale_reason = "The finalized environment is missing."
    elif value.get("env_spec_hash") != task_scene_hash(current_spec):
        stale_reason = "The environment physics changed after this task was defined."
    elif value.get("controller_version") != TASK_CONTROLLER_VERSION:
        stale_reason = "The task controller changed after this oracle was recorded."
    definition = {
        "schema_version": value.get("schema_version"),
        "env_id": value.get("env_id"),
        "instruction": value.get("instruction"),
        "max_steps": value.get("max_steps"),
        "tests": value.get("tests"),
    }
    if value.get("task_definition_hash") and value.get("task_definition_hash") != task_definition_hash(definition):
        stale_reason = "The trajectory tests changed after this oracle was recorded."
    status = str(value.get("status") or "error")
    if stale_reason and status in {"validated", "recording", "pending_oracle", "validation_failed"}:
        value["effective_status"] = "stale"
        value["stale_reason"] = stale_reason
    else:
        value["effective_status"] = status
        if status != "stale":
            value.pop("stale_reason", None)
    return value


def delete_task(scene_dir: Path, task_id: str) -> dict[str, Any]:
    task = read_task(scene_dir, task_id, include_staleness=False)
    shutil.rmtree(task_dir(scene_dir, task_id))
    return {"status": "success", "task_id": task["task_id"]}


def task_definition_hash(value: dict[str, Any]) -> str:
    definition = {
        "schema_version": value.get("schema_version"),
        "env_id": value.get("env_id"),
        "instruction": value.get("instruction"),
        "max_steps": value.get("max_steps"),
        "tests": value.get("tests"),
    }
    payload = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_scene_hash(spec: dict[str, Any]) -> str:
    """Hash physics and selector semantics while ignoring presentation-only edits."""

    objects = []
    for raw in spec.get("objects") or []:
        if not isinstance(raw, dict):
            continue
        objects.append(
            {
                key: raw.get(key)
                for key in (
                    "id",
                    "semantic_type",
                    "shape",
                    "body_type",
                    "position",
                    "size",
                    "yaw",
                    "tags",
                    "metadata",
                )
            }
        )
    value = {
        "schema_version": spec.get("schema_version"),
        "id": spec.get("id"),
        "world_size": spec.get("world_size"),
        "gravity": spec.get("gravity"),
        "objects": objects,
        "game": spec.get("game"),
        "mechanisms": spec.get("mechanisms") or [],
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_artifact_url(scene_dir: Path, path: Path) -> str:
    relative = path.resolve().relative_to(scene_dir.resolve())
    return f"/generated/{scene_dir.name}/{relative.as_posix()}?v={path.stat().st_mtime_ns}"


def new_attempt_id(scene_dir: Path, task_id: str) -> str:
    root = task_dir(scene_dir, task_id) / "attempts"
    index = len([path for path in root.iterdir() if path.is_dir()]) + 1 if root.is_dir() else 1
    return f"attempt-{index:04d}-{uuid.uuid4().hex[:8]}"


def _normalize_test(
    index: int,
    raw: Any,
    *,
    objects: list[Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TaskDefinitionError(f"test {index + 1} must be an object")
    mode = str(raw.get("mode") or "all").strip().lower()
    if mode not in SUPPORTED_TEST_MODES:
        raise TaskDefinitionError(f"test {index + 1} mode must be 'all' or 'any'")
    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise TaskDefinitionError(f"test {index + 1} needs at least one condition")
    if len(raw_conditions) > MAX_CONDITIONS_PER_TEST:
        raise TaskDefinitionError(
            f"test {index + 1} supports at most {MAX_CONDITIONS_PER_TEST} conditions"
        )
    conditions = [
        _normalize_condition(condition_index, condition, objects=objects, spec=spec)
        for condition_index, condition in enumerate(raw_conditions)
    ]
    condition_ids = [condition["id"] for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise TaskDefinitionError(f"test {index + 1} condition ids must be unique")
    ordered = [str(value).strip() for value in raw.get("ordered_condition_ids") or []]
    if len(ordered) != len(set(ordered)) or not set(ordered).issubset(set(condition_ids)):
        raise TaskDefinitionError(
            f"test {index + 1} ordered_condition_ids must be unique condition ids from that test"
        )
    if ordered and mode != "all":
        raise TaskDefinitionError("ordered conditions require test mode 'all'")
    for condition in conditions:
        if condition["id"] not in ordered:
            continue
        if issue := _ordered_condition_issue(condition):
            raise TaskDefinitionError(f"ordered condition {condition['id']!r} {issue}")
    return {
        "id": _normalize_id(raw.get("id") or f"test_{index + 1}"),
        "description": str(raw.get("description") or f"Task test {index + 1}").strip()[:600],
        "mode": mode,
        "conditions": conditions,
        "ordered_condition_ids": ordered,
        "source": str(raw.get("source") or "codex"),
    }


def _ordered_condition_issue(condition: dict[str, Any]) -> str:
    temporal = str(condition.get("temporal") or "eventually")
    if temporal in {"always", "never", "at_end"}:
        return (
            f"uses temporal operator {temporal!r}, which does not identify a chronological event; "
            "keep it as an unordered supporting condition"
        )

    predicate = condition.get("predicate") or {}
    predicate_type = str(predicate.get("type") or "")
    metric = str(predicate.get("metric") or "")
    is_trial_global_aggregate = (
        predicate_type == "displacement" and metric == "maximum"
    ) or (
        predicate_type == "axis_delta" and metric in {"maximum", "minimum"}
    )
    if is_trial_global_aggregate:
        return (
            f"uses trial-global aggregate {predicate_type} metric {metric!r}, whose value can be "
            "established before an earlier ordered phase; keep aggregate evidence unordered or "
            "replace it with a discrete overlap, contact, relation, or mechanism event"
        )
    return ""


def _system_safety_tests(
    *,
    objects: list[Any],
    spec: dict[str, Any],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_signatures = {
        (
            condition["temporal"],
            condition["predicate"]["type"],
            json.dumps(condition["predicate"], sort_keys=True, separators=(",", ":")),
        )
        for test in existing
        for condition in test.get("conditions") or []
    }
    raw_tests: list[dict[str, Any]] = []
    agent_selector = {"semantic_type": "agent"}
    raw_tests.append(
        {
            "id": "system_stay_in_bounds",
            "description": "The agent must remain inside the playable environment.",
            "mode": "all",
            "source": "system",
            "conditions": [
                {
                    "id": "always_in_bounds",
                    "description": "Agent remains within the play bounds.",
                    "temporal": "always",
                    "predicate": {"type": "in_bounds", "subject": agent_selector},
                }
            ],
            "ordered_condition_ids": [],
        }
    )
    values = []
    for raw in raw_tests:
        value = _normalize_test(len(existing) + len(values), raw, objects=objects, spec=spec)
        condition = value["conditions"][0]
        signature = (
            condition["temporal"],
            condition["predicate"]["type"],
            json.dumps(condition["predicate"], sort_keys=True, separators=(",", ":")),
        )
        if signature not in existing_signatures:
            values.append(value)
            existing_signatures.add(signature)
    return values


def _normalize_condition(
    index: int,
    raw: Any,
    *,
    objects: list[Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TaskDefinitionError(f"condition {index + 1} must be an object")
    temporal = str(raw.get("temporal") or "eventually").strip().lower()
    if temporal not in SUPPORTED_TEMPORAL_OPERATORS:
        raise TaskDefinitionError(
            f"condition {index + 1} has unsupported temporal operator {temporal!r}"
        )
    predicate = _normalize_predicate(raw.get("predicate"), objects=objects, spec=spec)
    condition_id = _normalize_id(raw.get("id") or f"condition_{index + 1}")
    value: dict[str, Any] = {
        "id": condition_id,
        "description": describe_assertion_condition(
            {**raw, "id": condition_id, "predicate": predicate}
        ),
        "temporal": temporal,
        "predicate": predicate,
    }
    if temporal == "sustained":
        value["frames"] = _bounded_int(
            raw.get("frames") if raw.get("frames") is not None else 30,
            "sustained frames",
            1,
            MAX_SUSTAINED_FRAMES,
        )
    if temporal == "count":
        minimum = _optional_non_negative_int(raw.get("min_count"), "min_count")
        maximum = _optional_non_negative_int(raw.get("max_count"), "max_count")
        if minimum is None and maximum is None:
            minimum = 1
        if minimum is not None:
            value["min_count"] = minimum
        if maximum is not None:
            value["max_count"] = maximum
        if minimum is not None and maximum is not None and minimum > maximum:
            raise TaskDefinitionError("min_count cannot exceed max_count")
    return value


def _normalize_predicate(raw: Any, *, objects: list[Any], spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TaskDefinitionError("condition predicate must be an object")
    predicate_type = str(raw.get("type") or "").strip().lower()
    if predicate_type not in SUPPORTED_PREDICATE_TYPES:
        raise TaskDefinitionError(
            f"unsupported task predicate {predicate_type!r}; expected one of {sorted(SUPPORTED_PREDICATE_TYPES)}"
        )
    result: dict[str, Any] = {"type": predicate_type}
    if predicate_type in {
        "overlap",
        "relation",
        "displacement",
        "axis_delta",
        "contact",
        "axis_value",
        "speed",
        "settled",
        "in_bounds",
        "grounded",
    }:
        result["subject"] = _normalize_resolved_selector(raw.get("subject"), "subject", objects)
        quantifier = str(raw.get("subject_quantifier") or "any").strip().lower()
        if quantifier not in {"any", "all"}:
            raise TaskDefinitionError("subject_quantifier must be 'any' or 'all'")
        result["subject_quantifier"] = quantifier
    if predicate_type in {"overlap", "relation", "contact"}:
        result["target"] = _normalize_resolved_selector(raw.get("target"), "target", objects)
    if predicate_type == "relation":
        relation = str(raw.get("relation") or "").strip().lower()
        if relation not in SUPPORTED_RELATIONS:
            raise TaskDefinitionError(f"unsupported task relation {relation!r}")
        result["relation"] = relation
        result["margin"] = _non_negative_float(
            raw.get("margin") if raw.get("margin") is not None else 0.0,
            "margin",
        )
        if relation == "near":
            result["max_distance"] = _positive_float(
                raw.get("max_distance") if raw.get("max_distance") is not None else 1.0,
                "max_distance",
            )
        if relation == "far_from":
            result["min_distance"] = _positive_float(
                raw.get("min_distance") if raw.get("min_distance") is not None else 1.0,
                "min_distance",
            )
    if predicate_type == "displacement":
        metric = str(raw.get("metric") or "maximum").strip().lower()
        space = str(raw.get("space") or "xy").strip().lower()
        if metric not in {"maximum", "final"}:
            raise TaskDefinitionError("displacement metric must be 'maximum' or 'final'")
        if space not in {"xy", "xyz"}:
            raise TaskDefinitionError("displacement space must be 'xy' or 'xyz'")
        result.update({"metric": metric, "space": space})
        _add_number_bounds(result, raw, default_min=0.25)
    elif predicate_type == "axis_delta":
        axis = str(raw.get("axis") or "z").strip().lower()
        metric = str(raw.get("metric") or "maximum").strip().lower()
        if axis not in {"x", "y", "z"}:
            raise TaskDefinitionError("axis_delta axis must be x, y, or z")
        if metric not in {"maximum", "minimum", "final"}:
            raise TaskDefinitionError(
                "axis_delta metric must be 'maximum', 'minimum', or 'final'"
            )
        result.update({"axis": axis, "metric": metric})
        _add_number_bounds(result, raw, default_min=0.25)
    elif predicate_type == "axis_value":
        axis = str(raw.get("axis") or "").strip().lower()
        if axis not in {"x", "y", "z"}:
            raise TaskDefinitionError("axis_value axis must be x, y, or z")
        result["axis"] = axis
        _add_number_bounds(result, raw, require=True)
    elif predicate_type == "speed":
        component = str(raw.get("component") or "linear").strip().lower()
        if component not in {"linear", "angular"}:
            raise TaskDefinitionError("speed component must be 'linear' or 'angular'")
        result["component"] = component
        _add_number_bounds(result, raw, require=True)
    elif predicate_type == "settled":
        result["max_linear_speed"] = _non_negative_float(
            raw.get("max_linear_speed") if raw.get("max_linear_speed") is not None else 0.12,
            "max_linear_speed",
        )
        result["max_angular_speed"] = _non_negative_float(
            raw.get("max_angular_speed") if raw.get("max_angular_speed") is not None else 0.25,
            "max_angular_speed",
        )
    elif predicate_type == "mechanism_state":
        mechanism_id = str(raw.get("mechanism_id") or "").strip()
        known = {str(item.get("id")) for item in spec.get("mechanisms") or [] if isinstance(item, dict)}
        if not mechanism_id or mechanism_id not in known:
            raise TaskDefinitionError("mechanism_state must reference an existing mechanism_id")
        state = str(raw.get("state") or "open").strip().lower()
        if state not in {"active", "open"}:
            raise TaskDefinitionError("mechanism_state state must be 'active' or 'open'")
        result.update({"mechanism_id": mechanism_id, "state": state})
        if state == "open":
            result["min_progress"] = _bounded_float(
                raw.get("min_progress") if raw.get("min_progress") is not None else 0.9,
                "min_progress",
                0.0,
                1.0,
            )
    elif predicate_type in {"jump_count", "step_count", "reset_count"}:
        _add_integer_bounds(result, raw, default_min=1)
    elif predicate_type == "reset_event":
        reason = str(raw.get("reason") or "any").strip().lower()
        if not reason:
            raise TaskDefinitionError("reset_event reason must be a non-empty string")
        result["reason"] = reason
    elif predicate_type == "terminal_event":
        event = str(raw.get("event") or "").strip().lower()
        if event not in SUPPORTED_TERMINAL_EVENTS:
            raise TaskDefinitionError(f"unsupported terminal event {event!r}")
        result["event"] = event
    return result


def _normalize_resolved_selector(raw: Any, name: str, objects: list[Any]) -> dict[str, Any]:
    try:
        selector = normalize_selector(raw, name)
    except Exception as exc:
        raise TaskDefinitionError(str(exc)) from exc
    matched = select_scene_objects(objects, selector)
    if not matched:
        raise TaskDefinitionError(f"{name} selector does not match any current scene object")
    return selector


def _condition_is_positive(condition: dict[str, Any]) -> bool:
    if condition["temporal"] in {"always", "never"}:
        return False
    predicate = condition["predicate"]
    if predicate["type"] == "terminal_event" and predicate.get("event") in {"hazard", "out_of_bounds"}:
        return False
    if condition["temporal"] == "count" and int(condition.get("min_count") or 0) <= 0:
        return False
    return True


def _add_number_bounds(
    target: dict[str, Any],
    raw: dict[str, Any],
    *,
    default_min: float | None = None,
    require: bool = False,
) -> None:
    minimum = _optional_float(raw.get("min_value", raw.get("min")), "min_value")
    maximum = _optional_float(raw.get("max_value", raw.get("max")), "max_value")
    if minimum is None and maximum is None:
        if require:
            raise TaskDefinitionError("predicate requires min_value or max_value")
        minimum = default_min
    if minimum is not None:
        target["min_value"] = minimum
    if maximum is not None:
        target["max_value"] = maximum
    if minimum is not None and maximum is not None and minimum > maximum:
        raise TaskDefinitionError("min_value cannot exceed max_value")


def _add_integer_bounds(target: dict[str, Any], raw: dict[str, Any], *, default_min: int) -> None:
    minimum = _optional_non_negative_int(raw.get("min_value", raw.get("min")), "min_value")
    maximum = _optional_non_negative_int(raw.get("max_value", raw.get("max")), "max_value")
    if minimum is None and maximum is None:
        minimum = default_min
    if minimum is not None:
        target["min_value"] = minimum
    if maximum is not None:
        target["max_value"] = maximum
    if minimum is not None and maximum is not None and minimum > maximum:
        raise TaskDefinitionError("min_value cannot exceed max_value")


def _load_final_spec(scene_dir: Path) -> dict[str, Any]:
    spec = _read_spec(scene_dir)
    if spec is None:
        raise TaskDefinitionError("finalize the environment before creating a task")
    return spec


def _read_spec(scene_dir: Path) -> dict[str, Any] | None:
    path = scene_dir / ENV_SPEC_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _require_controllable_agent(spec: dict[str, Any]) -> None:
    agents = [
        obj
        for obj in spec.get("objects") or []
        if isinstance(obj, dict) and obj.get("semantic_type") == "agent"
    ]
    if len(agents) != 1:
        raise TaskDefinitionError("tasks require exactly one controllable agent in the environment")


def _unique_task_id(scene_dir: Path, value: str) -> str:
    base = _slugify_task_id(value)[:48]
    candidate = base
    index = 2
    while task_path(scene_dir, candidate).exists():
        candidate = f"{base}_{index:02d}"
        index += 1
    return candidate


def _slugify_task_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    return (cleaned or "task")[:80]


def _validate_task_id(task_id: str) -> None:
    if not task_id or not TASK_ID_PATTERN.fullmatch(task_id):
        raise TaskDefinitionError("task id contains unsupported characters")


def _normalize_id(value: Any) -> str:
    result = _slugify_task_id(str(value or ""))
    if not result:
        raise TaskDefinitionError("id is required")
    return result


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskDefinitionError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise TaskDefinitionError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _optional_float(value, name)
    if result is None or result <= 0:
        raise TaskDefinitionError(f"{name} must be positive")
    return result


def _non_negative_float(value: Any, name: str) -> float:
    result = _optional_float(value, name)
    if result is None or result < 0:
        raise TaskDefinitionError(f"{name} must be non-negative")
    return result


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    result = _optional_float(value, name)
    if result is None or result < minimum or result > maximum:
        raise TaskDefinitionError(f"{name} must be between {minimum} and {maximum}")
    return result


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TaskDefinitionError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskDefinitionError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise TaskDefinitionError(f"{name} must be between {minimum} and {maximum}")
    return result


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _bounded_int(value, name, 0, MAX_TASK_STEPS)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
