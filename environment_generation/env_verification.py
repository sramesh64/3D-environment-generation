"""Prompt-derived deterministic checks for Environment Generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .mujoco_compile import compile_spec_to_mjcf
from .ramp_geometry import ramp_bounds, ramp_geometry_from_object, ramp_surface_height
from .scene_geometry import footprints_overlap, volume_contains
from .schema import EnvSpec3D, env_spec_to_dict, parse_env_spec_3d
from .studio_view_context import SCREEN_REGIONS, verification_camera, verification_projection


ENV_VERIFICATION_PLAN_FILENAME = "env_verification_plan.json"
ENV_VERIFICATION_REPORT_FILENAME = "env_verification_report.json"
ENV_VERIFICATION_SCHEMA_VERSION = "1.0"

SUPPORTED_CHECK_TYPES = {
    "object_count",
    "spatial_relation",
    "support_contact",
    "ramp_connection",
    "physics_probe",
    "screen_region",
    "screen_relation",
    "game_contract",
}
SUPPORTED_SEVERITIES = {"critical", "advisory"}
SUPPORTED_SPATIAL_RELATIONS = {
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "between",
    "near",
    "far_from",
    "inside",
    "on_surface",
}
SUPPORTED_SCREEN_RELATIONS = {"left_of", "right_of", "above", "below"}
SUPPORTED_PHYSICS_PROBES = {
    "passive_settles",
    "dynamic_object_stable",
    "pushable_moves",
}
SUPPORT_SURFACE_TYPES = {"ground", "platform", "ramp"}
PUSHABLE_TYPES = {"pushable_box", "ball", "cylinder"}


class EnvVerificationError(ValueError):
    """Raised when an environment verification plan is malformed."""


@dataclass(frozen=True)
class SceneObject3D:
    id: str
    semantic_type: str
    shape: str
    body_type: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float
    tags: frozenset[str]
    bounds: dict[str, float]
    visible: bool
    metadata: dict[str, Any]


def spec_hash(spec: dict[str, Any] | EnvSpec3D) -> str:
    if isinstance(spec, EnvSpec3D):
        value = env_spec_to_dict(spec)
    else:
        try:
            value = env_spec_to_dict(parse_env_spec_3d(spec))
        except Exception:
            value = spec
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_env_verification_plan(
    *,
    env_id: str,
    prompt: str,
    checks: Any,
    operation_count: int,
    draft_spec: dict[str, Any] | EnvSpec3D | None = None,
    draft_hash: str | None = None,
    screen_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(checks, list) or not checks:
        raise EnvVerificationError("checks must be a non-empty list")
    normalized_checks = [
        _normalize_check(index, check, screen_context=screen_context)
        for index, check in enumerate(checks)
    ]
    plan: dict[str, Any] = {
        "schema_version": ENV_VERIFICATION_SCHEMA_VERSION,
        "env_id": env_id,
        "prompt": prompt,
        "operation_count": int(operation_count),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": normalized_checks,
    }
    resolved_hash = draft_hash or (spec_hash(draft_spec) if draft_spec is not None else "")
    if resolved_hash:
        plan["draft_hash"] = resolved_hash
    return plan


def run_env_verification(
    *,
    env_id: str,
    plan: dict[str, Any],
    draft_spec: dict[str, Any] | EnvSpec3D,
    final_spec: dict[str, Any] | EnvSpec3D | None,
    operation_count: int,
    readiness_errors: list[str] | None = None,
) -> dict[str, Any]:
    checks = plan.get("checks")
    if not isinstance(checks, list):
        raise EnvVerificationError("verification plan is missing checks")
    draft_json = _spec_json(draft_spec)
    final_json = _spec_json(final_spec) if final_spec is not None else None
    current_hash = spec_hash(draft_json)
    try:
        objects = scene_objects(draft_json)
    except Exception as exc:
        objects = []
        inspect_error = str(exc)
    else:
        inspect_error = ""

    results: list[dict[str, Any]] = []
    for check in checks:
        check_type = str(check.get("type") or "")
        if inspect_error and check_type not in {"physics_probe", "game_contract"}:
            result = _check_result(
                check,
                passed=False,
                message=f"Scene geometry could not be inspected: {inspect_error}",
                repair_hints=["Fix schema or geometry errors before running environment checks."],
            )
        elif check_type == "object_count":
            result = _evaluate_object_count(check, objects)
        elif check_type == "spatial_relation":
            result = _evaluate_spatial_relation(check, objects)
        elif check_type == "support_contact":
            result = _evaluate_support_contact(check, objects)
        elif check_type == "ramp_connection":
            result = _evaluate_ramp_connection(check, objects)
        elif check_type == "screen_region":
            result = _evaluate_screen_region(check, objects)
        elif check_type == "screen_relation":
            result = _evaluate_screen_relation(check, objects)
        elif check_type == "physics_probe":
            result = _evaluate_physics_probe(
                check,
                final_spec=final_json,
                readiness_errors=readiness_errors or [],
            )
        elif check_type == "game_contract":
            result = _evaluate_game_contract(check, draft_json)
        else:
            result = _check_result(
                check,
                passed=False,
                message=f"Unsupported check type {check_type!r}.",
                repair_hints=["Replace the check with a supported 3D environment verification check."],
            )
        results.append(result)

    critical_failures = sum(
        1 for result in results if result["severity"] == "critical" and not result["passed"]
    )
    advisory_failures = sum(
        1 for result in results if result["severity"] == "advisory" and not result["passed"]
    )
    passed = sum(1 for result in results if result["passed"])
    if critical_failures:
        status = "failed"
    elif advisory_failures:
        status = "advisory_issues"
    else:
        status = "passed"
    return {
        "schema_version": ENV_VERIFICATION_SCHEMA_VERSION,
        "env_id": env_id,
        "plan_created_at": plan.get("created_at"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation_count": int(operation_count),
        "plan_draft_hash": plan.get("draft_hash") or "",
        "draft_hash": current_hash,
        "status": status,
        "blocking": bool(critical_failures),
        "summary": {
            "total": len(results),
            "passed": passed,
            "critical_failures": critical_failures,
            "advisory_failures": advisory_failures,
        },
        "results": results,
        "next_action": _report_next_action(critical_failures, advisory_failures),
    }


def load_env_verification_plan(scene_dir: Path) -> dict[str, Any] | None:
    return _read_optional_json(scene_dir / ENV_VERIFICATION_PLAN_FILENAME)


def load_env_verification_report(scene_dir: Path) -> dict[str, Any] | None:
    return _read_optional_json(scene_dir / ENV_VERIFICATION_REPORT_FILENAME)


def write_env_verification_plan(scene_dir: Path, plan: dict[str, Any]) -> Path:
    path = scene_dir / ENV_VERIFICATION_PLAN_FILENAME
    _atomic_write_json(path, plan)
    return path


def write_env_verification_report(scene_dir: Path, report: dict[str, Any]) -> Path:
    path = scene_dir / ENV_VERIFICATION_REPORT_FILENAME
    _atomic_write_json(path, report)
    return path


def clear_env_verification_report(scene_dir: Path) -> None:
    path = scene_dir / ENV_VERIFICATION_REPORT_FILENAME
    if path.is_file():
        path.unlink()


def env_verification_summary(
    scene_dir: Path,
    *,
    draft_hash: str | None = None,
    operation_count: int | None = None,
) -> dict[str, Any]:
    plan = load_env_verification_plan(scene_dir)
    report = load_env_verification_report(scene_dir)
    if not plan:
        return {
            "status": "missing",
            "has_plan": False,
            "has_report": False,
            "label": "Env checks: not defined",
            "critical_failures": 0,
            "advisory_failures": 0,
        }
    critical_checks = sum(
        1
        for check in plan.get("checks") or []
        if isinstance(check, dict) and check.get("severity") == "critical"
    )
    if not report:
        return {
            "status": "not_run",
            "has_plan": True,
            "has_report": False,
            "critical_checks": critical_checks,
            "label": "Env checks: not run",
            "critical_failures": 0,
            "advisory_failures": 0,
        }
    stale = _report_is_stale(report, draft_hash=draft_hash, operation_count=operation_count)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    critical_failures = int(summary.get("critical_failures") or 0)
    advisory_failures = int(summary.get("advisory_failures") or 0)
    if stale:
        status = "stale"
        label = "Env checks: stale"
    elif critical_failures:
        status = "critical_failures"
        label = f"Env checks: {critical_failures} critical failure"
        if critical_failures != 1:
            label += "s"
    elif advisory_failures:
        status = "advisory_issues"
        label = "Env checks: advisory issues only"
    else:
        status = "passed"
        label = "Env checks: passed"
    return {
        "status": status,
        "has_plan": True,
        "has_report": True,
        "stale": stale,
        "critical_checks": critical_checks,
        "critical_failures": critical_failures,
        "advisory_failures": advisory_failures,
        "label": label,
        "report_operation_count": _optional_int(report.get("operation_count")),
        "draft_hash": report.get("draft_hash") or "",
    }


def finalization_block(
    *,
    scene_dir: Path,
    draft_hash: str,
    operation_count: int,
) -> dict[str, Any] | None:
    plan = load_env_verification_plan(scene_dir)
    if not plan:
        return None
    critical_checks = [
        check
        for check in plan.get("checks") or []
        if isinstance(check, dict) and check.get("severity") == "critical"
    ]
    if not critical_checks:
        return None
    report = load_env_verification_report(scene_dir)
    if not report:
        return {
            "reason": "missing_report",
            "summary": env_verification_summary(
                scene_dir,
                draft_hash=draft_hash,
                operation_count=operation_count,
            ),
            "message": "Critical environment checks exist but run_env_verification has not been run.",
            "next_action": "Run run_env_verification, repair critical failures, then finalize again.",
        }
    if _report_is_stale(report, draft_hash=draft_hash, operation_count=operation_count):
        return {
            "reason": "stale_report",
            "summary": env_verification_summary(
                scene_dir,
                draft_hash=draft_hash,
                operation_count=operation_count,
            ),
            "message": "The latest environment verification report is stale for the current draft.",
            "next_action": "Run run_env_verification on the current draft before finalizing.",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    critical_failures = int(summary.get("critical_failures") or 0)
    if critical_failures:
        return {
            "reason": "critical_failures",
            "summary": env_verification_summary(
                scene_dir,
                draft_hash=draft_hash,
                operation_count=operation_count,
            ),
            "message": f"{critical_failures} critical environment verification check failed.",
            "report": report,
            "next_action": "Repair the critical failures and rerun run_env_verification.",
        }
    return None


def scene_objects(spec: dict[str, Any] | EnvSpec3D) -> list[SceneObject3D]:
    parsed = spec if isinstance(spec, EnvSpec3D) else parse_env_spec_3d(spec)
    objects: list[SceneObject3D] = []
    for obj in parsed.objects:
        obj_json = obj.model_dump(mode="json")
        if obj.shape == "ramp":
            geometry = ramp_geometry_from_object(obj_json)
            position = geometry.center
            bounds = ramp_bounds(geometry)
        else:
            position = _tuple3(obj.position)
            bounds = _object_bounds(obj_json)
        objects.append(
            SceneObject3D(
                id=obj.id,
                semantic_type=obj.semantic_type.lower(),
                shape=obj.shape.lower(),
                body_type=obj.body_type.lower(),
                position=position,
                size=_tuple3(obj.size),
                yaw=float(obj.yaw or 0.0),
                tags=frozenset(
                    {
                        obj.semantic_type.lower(),
                        *(str(tag).lower() for tag in obj.tags),
                    }
                ),
                bounds=bounds,
                visible=bool(obj.visible),
                metadata=dict(obj.metadata),
            )
        )
    return objects


def normalize_selector(raw: Any, name: str = "selector") -> dict[str, Any]:
    """Normalize the selector language shared by static and behavioral checks."""

    return _normalize_selector(raw, name)


def select_scene_objects(
    objects: list[SceneObject3D],
    selector: dict[str, Any],
) -> list[SceneObject3D]:
    return _select_objects(objects, selector)


def scene_object_at(
    obj: SceneObject3D,
    position: tuple[float, float, float] | list[float],
    *,
    yaw: float | None = None,
) -> SceneObject3D:
    """Return a verifier object at a simulated world pose."""

    x, y, z = (float(value) for value in position)
    resolved_yaw = obj.yaw if yaw is None else float(yaw)
    if yaw is not None and not math.isclose(resolved_yaw, obj.yaw, abs_tol=1e-12):
        mapping = _scene_object_mapping(obj)
        mapping["position"] = [x, y, z]
        mapping["yaw"] = resolved_yaw
        return replace(
            obj,
            position=(x, y, z),
            yaw=resolved_yaw,
            bounds=_object_bounds(mapping),
        )
    dx = x - obj.position[0]
    dy = y - obj.position[1]
    dz = z - obj.position[2]
    bounds = {
        "x1": obj.bounds["x1"] + dx,
        "x2": obj.bounds["x2"] + dx,
        "y1": obj.bounds["y1"] + dy,
        "y2": obj.bounds["y2"] + dy,
        "z1": obj.bounds["z1"] + dz,
        "z2": obj.bounds["z2"] + dz,
    }
    return replace(obj, position=(x, y, z), yaw=resolved_yaw, bounds=bounds)


def spatial_relation_holds(
    subject: SceneObject3D,
    target: SceneObject3D,
    *,
    relation: str,
    margin: float = 0.0,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> bool:
    check: dict[str, Any] = {
        "relation": relation,
        "margin": float(margin),
    }
    if min_distance is not None:
        check["min_distance"] = float(min_distance)
    if max_distance is not None:
        check["max_distance"] = float(max_distance)
    return _relation_holds(subject, target, check)


def _normalize_check(index: int, raw: Any, *, screen_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EnvVerificationError(f"check {index + 1} must be an object")
    check_type = str(raw.get("type") or "").strip().lower()
    if check_type not in SUPPORTED_CHECK_TYPES:
        raise EnvVerificationError(
            f"check {index + 1} has unsupported type {check_type!r}; "
            f"expected one of {sorted(SUPPORTED_CHECK_TYPES)}"
        )
    severity = str(raw.get("severity") or "critical").strip().lower()
    if severity not in SUPPORTED_SEVERITIES:
        raise EnvVerificationError(
            f"check {index + 1} has unsupported severity {severity!r}; use 'critical' or 'advisory'"
        )
    normalized = {
        "id": _normalize_id(raw.get("id") or f"{check_type}_{index + 1}"),
        "type": check_type,
        "severity": severity,
        "description": str(raw.get("description") or raw.get("intent") or "").strip(),
    }
    if check_type == "object_count":
        normalized.update(_normalize_object_count_check(raw))
    elif check_type == "spatial_relation":
        normalized.update(_normalize_spatial_relation_check(raw))
    elif check_type == "support_contact":
        normalized.update(_normalize_support_contact_check(raw))
    elif check_type == "ramp_connection":
        normalized.update(_normalize_ramp_connection_check(raw))
    elif check_type == "physics_probe":
        normalized.update(_normalize_physics_probe_check(raw))
    elif check_type == "screen_region":
        normalized.update(_normalize_screen_region_check(raw, screen_context=screen_context))
    elif check_type == "screen_relation":
        normalized.update(_normalize_screen_relation_check(raw, screen_context=screen_context))
    return normalized


def _evaluate_game_contract(check: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    from .courtyard import validate_courtyard_layout

    try:
        issues = validate_courtyard_layout(spec)
    except Exception as exc:
        issues = [str(exc)]
    return _check_result(
        check,
        passed=not issues,
        message="Robot courtyard game contract is valid." if not issues else "Robot courtyard game contract failed validation.",
        metrics={"issues": issues},
        repair_hints=[] if not issues else issues[:8],
    )


def _normalize_object_count_check(raw: dict[str, Any]) -> dict[str, Any]:
    selector = raw.get("selector")
    if selector is None:
        selector = {key: raw[key] for key in ("id", "semantic_type", "object_type", "body_type", "shape", "tag") if key in raw}
    result: dict[str, Any] = {"selector": _normalize_selector(selector, "selector")}
    exact = raw.get("exact")
    minimum = raw.get("min", raw.get("minimum"))
    maximum = raw.get("max", raw.get("maximum"))
    if exact is None and minimum is None and maximum is None:
        minimum = 1
    if exact is not None:
        result["exact"] = _non_negative_int(exact, "exact")
    if minimum is not None:
        result["min"] = _non_negative_int(minimum, "min")
    if maximum is not None:
        result["max"] = _non_negative_int(maximum, "max")
    if "min" in result and "max" in result and result["min"] > result["max"]:
        raise EnvVerificationError("object_count min cannot exceed max")
    return result


def _normalize_spatial_relation_check(raw: dict[str, Any]) -> dict[str, Any]:
    relation = str(raw.get("relation") or "").strip().lower()
    if relation not in SUPPORTED_SPATIAL_RELATIONS:
        raise EnvVerificationError(
            f"spatial_relation has unsupported relation {relation!r}; "
            f"expected one of {sorted(SUPPORTED_SPATIAL_RELATIONS)}"
        )
    result: dict[str, Any] = {
        "subject": _normalize_selector(raw.get("subject"), "subject"),
        "relation": relation,
        "margin": float(raw.get("margin", 0.0) or 0.0),
    }
    if relation == "between":
        targets = raw.get("targets", raw.get("target"))
        if targets is None and ("left" in raw or "right" in raw):
            targets = [raw.get("left"), raw.get("right")]
        if not isinstance(targets, list) or len(targets) != 2:
            raise EnvVerificationError("between relation requires two targets")
        result["targets"] = [
            _normalize_selector(targets[0], "targets[0]"),
            _normalize_selector(targets[1], "targets[1]"),
        ]
    else:
        result["target"] = _normalize_selector(raw.get("target"), "target")
    if relation == "near":
        result["max_distance"] = _positive_float(
            raw.get("max_distance", raw.get("distance", 2.0)),
            "max_distance",
        )
    if relation == "far_from":
        result["min_distance"] = _positive_float(
            raw.get("min_distance", raw.get("distance", 2.0)),
            "min_distance",
        )
    return result


def _normalize_support_contact_check(raw: dict[str, Any]) -> dict[str, Any]:
    selector = raw.get("selector")
    if selector is None:
        selector = {key: raw[key] for key in ("id", "semantic_type", "object_type", "body_type", "shape", "tag") if key in raw}
    if not selector:
        selector = {"body_type": "dynamic"}
    return {
        "selector": _normalize_selector(selector, "selector"),
        "surface_selector": _normalize_selector(
            raw.get("surface_selector")
            or raw.get("surface")
            or {"semantic_type": sorted(SUPPORT_SURFACE_TYPES)},
            "surface_selector",
        ),
        "max_gap": float(raw.get("max_gap", 0.18) or 0.18),
        "penetration_tolerance": float(raw.get("penetration_tolerance", 0.08) or 0.08),
        "min_xy_overlap": float(raw.get("min_xy_overlap", 0.04) or 0.04),
    }


def _normalize_ramp_connection_check(raw: dict[str, Any]) -> dict[str, Any]:
    max_horizontal_gap = float(raw.get("max_horizontal_gap", 0.2))
    max_vertical_gap = float(raw.get("max_vertical_gap", 0.08))
    if max_horizontal_gap < 0 or max_vertical_gap < 0:
        raise EnvVerificationError("ramp_connection gap tolerances must be non-negative")
    return {
        "ramp": _normalize_selector(raw.get("ramp", raw.get("selector", "ramp")), "ramp"),
        "low_surface": _normalize_selector(raw.get("low_surface"), "low_surface"),
        "high_surface": _normalize_selector(raw.get("high_surface"), "high_surface"),
        "max_horizontal_gap": max_horizontal_gap,
        "max_vertical_gap": max_vertical_gap,
    }


def _normalize_physics_probe_check(raw: dict[str, Any]) -> dict[str, Any]:
    probe = str(raw.get("probe") or "").strip()
    if probe not in SUPPORTED_PHYSICS_PROBES:
        raise EnvVerificationError(
            f"physics_probe has unsupported probe {probe!r}; expected one of {sorted(SUPPORTED_PHYSICS_PROBES)}"
        )
    result: dict[str, Any] = {"probe": probe}
    if raw.get("object_id") is not None:
        result["object_id"] = str(raw["object_id"])
    return result


def _normalize_screen_region_check(
    raw: dict[str, Any],
    *,
    screen_context: dict[str, Any] | None,
) -> dict[str, Any]:
    subject = raw.get("subject", raw.get("selector"))
    region = str(raw.get("region") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if region not in SCREEN_REGIONS:
        raise EnvVerificationError(
            f"screen_region has unsupported region {region!r}; expected one of {sorted(SCREEN_REGIONS)}"
        )
    projection = verification_projection(screen_context, region)
    if projection is None:
        raise EnvVerificationError(
            "screen_region requires the exact submitted Studio camera context for this revision"
        )
    return {
        "subject": _normalize_selector(subject, "subject"),
        "region": region,
        "projection": projection,
    }


def _normalize_screen_relation_check(
    raw: dict[str, Any],
    *,
    screen_context: dict[str, Any] | None,
) -> dict[str, Any]:
    relation = str(raw.get("relation") or "").strip().lower()
    if relation not in SUPPORTED_SCREEN_RELATIONS:
        raise EnvVerificationError(
            f"screen_relation has unsupported relation {relation!r}; expected one of {sorted(SUPPORTED_SCREEN_RELATIONS)}"
        )
    projection = verification_camera(screen_context)
    if projection is None:
        raise EnvVerificationError(
            "screen_relation requires the exact submitted Studio camera context for this revision"
        )
    return {
        "subject": _normalize_selector(raw.get("subject"), "subject"),
        "target": _normalize_selector(raw.get("target"), "target"),
        "relation": relation,
        "margin_uv": max(0.0, float(raw.get("margin_uv", 0.02) or 0.02)),
        "projection": projection,
    }


def _normalize_selector(raw: Any, name: str) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"semantic_type": raw}
    if not isinstance(raw, dict):
        raise EnvVerificationError(f"{name} must be a selector object or semantic type string")
    aliases = dict(raw)
    if "object_type" in aliases and "semantic_type" not in aliases:
        aliases["semantic_type"] = aliases.pop("object_type")
    if "body_id" in aliases and "id" not in aliases:
        aliases["id"] = aliases.pop("body_id")
    supported = {"id", "semantic_type", "body_type", "shape", "tag"}
    selector = {key: aliases[key] for key in supported if key in aliases and aliases[key] is not None}
    if not selector:
        label = "selector" if name == "selector" else f"{name} selector"
        received = ", ".join(sorted(str(key) for key in raw.keys())) or "none"
        raise EnvVerificationError(
            f"{label} must include id, semantic_type, object_type, body_type, shape, or tag; got keys: {received}"
        )
    return selector


def _evaluate_object_count(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    matched = _select_objects(objects, check["selector"])
    count = len(matched)
    failures = []
    if "exact" in check and count != check["exact"]:
        failures.append(f"expected exactly {check['exact']}, found {count}")
    if "min" in check and count < check["min"]:
        failures.append(f"expected at least {check['min']}, found {count}")
    if "max" in check and count > check["max"]:
        failures.append(f"expected at most {check['max']}, found {count}")
    passed = not failures
    return _check_result(
        check,
        passed=passed,
        message=(
            f"Found {count} matching object(s)."
            if passed
            else "Object count failed: " + "; ".join(failures) + "."
        ),
        metrics={"count": count, "matched_ids": [item.id for item in matched]},
        repair_hints=[] if passed else [_count_repair_hint(check, count)],
    )


def _evaluate_spatial_relation(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    subjects = _select_objects(objects, check["subject"])
    if not subjects:
        return _check_result(
            check,
            passed=False,
            message="No subject object matched the spatial_relation selector.",
            repair_hints=["Add or retag the subject object, or update the selector."],
        )
    relation = check["relation"]
    if relation == "between":
        target_a = _select_objects(objects, check["targets"][0])
        target_b = _select_objects(objects, check["targets"][1])
        if not target_a or not target_b:
            return _check_result(
                check,
                passed=False,
                message="The between relation target selectors did not both match objects.",
                repair_hints=["Place the subject between two existing target objects or adjust selectors."],
            )
        passed = any(
            _is_between(subject, a, b, margin=check["margin"])
            for subject in subjects
            for a in target_a
            for b in target_b
        )
        target_ids = [item.id for item in target_a + target_b]
        distances: dict[str, float] = {}
    else:
        targets = _select_objects(objects, check["target"])
        if not targets:
            return _check_result(
                check,
                passed=False,
                message="No target object matched the spatial_relation selector.",
                repair_hints=["Add or retag the target object, or update the selector."],
            )
        passed = any(
            _relation_holds(subject, target, check)
            for subject in subjects
            for target in targets
        )
        target_ids = [item.id for item in targets]
        distances = {
            f"{subject.id}->{target.id}": round(_distance(_center(subject), _center(target)), 4)
            for subject in subjects
            for target in targets
            if relation in {"near", "far_from"}
        }
    return _check_result(
        check,
        passed=passed,
        message=(
            f"Spatial relation {relation!r} is satisfied."
            if passed
            else f"Spatial relation {relation!r} was not satisfied."
        ),
        metrics={
            "subject_ids": [item.id for item in subjects],
            "target_ids": target_ids,
            "distances": distances,
        },
        repair_hints=[] if passed else [_spatial_repair_hint(relation)],
    )


def _evaluate_support_contact(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    subjects = _select_objects(objects, check["selector"])
    surfaces = [
        item
        for item in _select_objects(objects, check["surface_selector"])
        if item.body_type == "static"
    ]
    if not subjects:
        return _check_result(
            check,
            passed=False,
            message="No object matched the support_contact selector.",
            repair_hints=["Add or retag the object that needs support."],
        )
    if not surfaces:
        return _check_result(
            check,
            passed=False,
            message="No static support surface matched the support_contact surface selector.",
            repair_hints=["Add a ground, platform, or ramp support surface."],
        )
    failures: list[dict[str, Any]] = []
    supported_ids: list[str] = []
    for subject in subjects:
        support = _best_support(subject, surfaces)
        if support is None:
            failures.append({"id": subject.id, "issue": "no_xy_overlap"})
            continue
        surface, overlap_area, vertical_gap = support
        if overlap_area < check["min_xy_overlap"]:
            failures.append({"id": subject.id, "surface_id": surface.id, "issue": "insufficient_xy_overlap"})
            continue
        if vertical_gap > check["max_gap"]:
            failures.append(
                {
                    "id": subject.id,
                    "surface_id": surface.id,
                    "issue": "floating",
                    "vertical_gap": round(vertical_gap, 4),
                }
            )
            continue
        if vertical_gap < -check["penetration_tolerance"]:
            failures.append(
                {
                    "id": subject.id,
                    "surface_id": surface.id,
                    "issue": "clipping",
                    "vertical_gap": round(vertical_gap, 4),
                }
            )
            continue
        supported_ids.append(subject.id)
    passed = not failures
    return _check_result(
        check,
        passed=passed,
        message=(
            "All selected objects are supported by a static surface."
            if passed
            else "Support contact failed for one or more selected objects."
        ),
        metrics={
            "checked_ids": [item.id for item in subjects],
            "supported_ids": supported_ids,
            "surface_ids": [item.id for item in surfaces],
            "failures": failures,
        },
        repair_hints=[] if passed else ["Move selected objects onto a ground, platform, or ramp without floating or clipping."],
    )


def _evaluate_ramp_connection(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    ramps = [obj for obj in _select_objects(objects, check["ramp"]) if obj.shape == "ramp"]
    low_surfaces = _select_objects(objects, check["low_surface"])
    high_surfaces = _select_objects(objects, check["high_surface"])
    if not ramps:
        return _check_result(
            check,
            passed=False,
            message="No ramp matched the ramp_connection selector.",
            repair_hints=["Add the requested ramp or correct the ramp selector."],
        )
    if not low_surfaces or not high_surfaces:
        return _check_result(
            check,
            passed=False,
            message="The ramp_connection surface selectors did not both match objects.",
            metrics={
                "low_surface_ids": [surface.id for surface in low_surfaces],
                "high_surface_ids": [surface.id for surface in high_surfaces],
            },
            repair_hints=["Select the intended ground/platform at both ends of the ramp."],
        )

    measurements: list[dict[str, Any]] = []
    passed = False
    for ramp in ramps:
        geometry = ramp_geometry_from_object(_scene_object_mapping(ramp))
        low_match = _best_endpoint_surface_match(geometry.low_end, low_surfaces)
        high_match = _best_endpoint_surface_match(geometry.high_end, high_surfaces)
        low_ok = _endpoint_match_passes(low_match, check)
        high_ok = _endpoint_match_passes(high_match, check)
        passed = passed or (low_ok and high_ok)
        measurements.append(
            {
                "ramp_id": ramp.id,
                "low_end": [round(value, 4) for value in geometry.low_end],
                "high_end": [round(value, 4) for value in geometry.high_end],
                "low_surface": low_match,
                "high_surface": high_match,
                "low_connected": low_ok,
                "high_connected": high_ok,
                "slope_degrees": round(math.degrees(geometry.angle), 3),
            }
        )
    return _check_result(
        check,
        passed=passed,
        message=(
            "A ramp connects the selected low and high walkable surfaces."
            if passed
            else "No ramp connects both selected walkable surfaces within tolerance."
        ),
        metrics={
            "ramps": measurements,
            "max_horizontal_gap": check["max_horizontal_gap"],
            "max_vertical_gap": check["max_vertical_gap"],
        },
        repair_hints=(
            []
            if passed
            else [
                "Use set_ramp_geometry so the low endpoint touches the lower surface and the high endpoint touches the upper surface."
            ]
        ),
    )


def _evaluate_screen_region(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    subjects = _select_objects(objects, check["subject"])
    if not subjects:
        return _check_result(
            check,
            passed=False,
            message="No object matched the screen_region subject selector.",
            repair_hints=["Add or retag the requested object, or correct the screen_region selector."],
        )
    projection = check["projection"]
    bounds = projection["region_bounds_uv"]
    projected = []
    for subject in subjects:
        uv, depth = _project_world_to_screen(_center(subject), projection["camera"])
        visible = depth > 0 and 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1
        inside = visible and (
            float(bounds["left"]) <= uv[0] <= float(bounds["right"])
            and float(bounds["top"]) <= uv[1] <= float(bounds["bottom"])
        )
        projected.append(
            {
                "id": subject.id,
                "screen_uv": [round(uv[0], 4), round(uv[1], 4)],
                "camera_depth": round(depth, 4),
                "visible": visible,
                "inside_region": inside,
            }
        )
    passed = any(item["inside_region"] for item in projected)
    anchor = projection.get("anchor_world_position")
    repair = f"Move the subject into the submitted view's {check['region'].replace('_', '-')} region."
    if isinstance(anchor, list) and len(anchor) == 3:
        repair += f" A suitable world-space anchor from that exact view is {[round(float(value), 3) for value in anchor]}."
    return _check_result(
        check,
        passed=passed,
        message=(
            f"At least one subject appears in the submitted view's {check['region'].replace('_', '-')} region."
            if passed
            else f"No subject appears in the submitted view's {check['region'].replace('_', '-')} region."
        ),
        metrics={
            "region": check["region"],
            "region_bounds_uv": bounds,
            "context_review_id": projection.get("context_review_id") or "",
            "subjects": projected,
        },
        repair_hints=[] if passed else [repair],
    )


def _evaluate_screen_relation(check: dict[str, Any], objects: list[SceneObject3D]) -> dict[str, Any]:
    subjects = _select_objects(objects, check["subject"])
    targets = _select_objects(objects, check["target"])
    if not subjects or not targets:
        return _check_result(
            check,
            passed=False,
            message="The screen_relation subject or target selector did not match an object.",
            repair_hints=["Add or retag the requested objects, or correct the screen_relation selectors."],
        )
    camera = check["projection"]["camera"]
    margin = float(check.get("margin_uv") or 0.0)
    subject_points = {item.id: _project_world_to_screen(_center(item), camera) for item in subjects}
    target_points = {item.id: _project_world_to_screen(_center(item), camera) for item in targets}
    pairs = []
    for subject in subjects:
        subject_uv, subject_depth = subject_points[subject.id]
        for target in targets:
            target_uv, target_depth = target_points[target.id]
            visible = (
                subject_depth > 0
                and target_depth > 0
                and all(0 <= value <= 1 for value in (*subject_uv, *target_uv))
            )
            relation_holds = visible and _screen_relation_holds(subject_uv, target_uv, check["relation"], margin)
            pairs.append(
                {
                    "subject_id": subject.id,
                    "target_id": target.id,
                    "subject_uv": [round(value, 4) for value in subject_uv],
                    "target_uv": [round(value, 4) for value in target_uv],
                    "visible": visible,
                    "relation_holds": relation_holds,
                }
            )
    passed = any(pair["relation_holds"] for pair in pairs)
    direction = check["relation"].replace("_", " ")
    return _check_result(
        check,
        passed=passed,
        message=(
            f"The submitted-view screen relation {check['relation']!r} is satisfied."
            if passed
            else f"The subject is not visibly {direction} the target in the submitted view."
        ),
        metrics={
            "relation": check["relation"],
            "margin_uv": margin,
            "context_review_id": check["projection"].get("context_review_id") or "",
            "pairs": pairs,
        },
        repair_hints=[] if passed else [f"Move the subject visibly {direction} the target in the submitted camera view."],
    )


def _screen_relation_holds(
    subject_uv: tuple[float, float],
    target_uv: tuple[float, float],
    relation: str,
    margin: float,
) -> bool:
    if relation == "left_of":
        return subject_uv[0] <= target_uv[0] - margin
    if relation == "right_of":
        return subject_uv[0] >= target_uv[0] + margin
    if relation == "above":
        return subject_uv[1] <= target_uv[1] - margin
    if relation == "below":
        return subject_uv[1] >= target_uv[1] + margin
    return False


def _project_world_to_screen(
    world_point: tuple[float, float, float],
    camera: dict[str, Any],
) -> tuple[tuple[float, float], float]:
    position = tuple(float(value) for value in camera["position"])
    target = tuple(float(value) for value in camera["target"])
    forward = _normalize_vector(tuple(target[index] - position[index] for index in range(3)))
    right = _normalize_vector((-forward[1], forward[0], 0.0))
    up = _cross(forward, right)
    relative = tuple(float(world_point[index]) - position[index] for index in range(3))
    depth = _dot(relative, forward)
    if depth <= 1e-9:
        return ((float("inf"), float("inf")), depth)
    x_camera = _dot(relative, right)
    y_camera = _dot(relative, up)
    tangent = math.tan(math.radians(float(camera["fov_y_degrees"])) * 0.5)
    ndc_x = x_camera / (depth * tangent * float(camera["aspect"]))
    ndc_y = y_camera / (depth * tangent)
    return ((ndc_x + 1.0) * 0.5, (1.0 - ndc_y) * 0.5), depth


def _normalize_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(value, value))
    if length <= 1e-12:
        raise EnvVerificationError("submitted camera has a degenerate viewing direction")
    return tuple(item / length for item in value)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _evaluate_physics_probe(
    check: dict[str, Any],
    *,
    final_spec: dict[str, Any] | None,
    readiness_errors: list[str],
) -> dict[str, Any]:
    if final_spec is None:
        return _check_result(
            check,
            passed=False,
            message="Physics probe cannot run until the draft is finalizable.",
            metrics={"readiness_errors": readiness_errors},
            repair_hints=["Fix readiness errors before running deterministic MuJoCo probes."],
        )
    try:
        if check["probe"] == "passive_settles":
            probe_result = _probe_passive_settles(final_spec)
        elif check["probe"] == "dynamic_object_stable":
            probe_result = _probe_dynamic_object_stable(final_spec, object_id=check.get("object_id"))
        elif check["probe"] == "pushable_moves":
            probe_result = _probe_pushable_moves(final_spec, object_id=check.get("object_id"))
        else:
            raise EnvVerificationError(f"unsupported physics probe {check['probe']!r}")
    except Exception as exc:
        return _check_result(
            check,
            passed=False,
            message=f"Physics probe {check['probe']!r} could not run: {exc}",
            repair_hints=["Fix the environment shape or probe target, then rerun verification."],
        )
    passed = bool(probe_result.get("passed"))
    return _check_result(
        check,
        passed=passed,
        message=(
            f"Physics probe {check['probe']!r} passed."
            if passed
            else f"Physics probe {check['probe']!r} failed."
        ),
        metrics=probe_result,
        repair_hints=[] if passed else [_physics_repair_hint(check["probe"], probe_result)],
    )


def _probe_passive_settles(spec: dict[str, Any]) -> dict[str, Any]:
    mujoco, model, data = _load_mujoco_model(spec)
    targets = _dynamic_probe_objects(spec)
    initial = _body_positions(mujoco, model, data, targets)
    max_speed = 0.0
    min_z = math.inf
    for _ in range(240):
        mujoco.mj_step(model, data)
        if not _data_finite(data):
            return {"probe": "passive_settles", "passed": False, "issue": "simulation became non-finite"}
        max_speed = max(max_speed, _max_body_speed(mujoco, model, data, targets))
        for pos in _body_positions(mujoco, model, data, targets).values():
            min_z = min(min_z, pos[2])
    final = _body_positions(mujoco, model, data, targets)
    max_displacement = _max_displacement(initial, final)
    passed = max_speed < 80.0 and (min_z == math.inf or min_z > -5.0)
    return {
        "probe": "passive_settles",
        "passed": passed,
        "body_ids": [item.id for item in targets],
        "max_speed": round(max_speed, 4),
        "max_displacement": round(max_displacement, 4),
        "min_z": None if min_z == math.inf else round(min_z, 4),
        "issue": "" if passed else "Dynamic bodies became unstable or fell far below the scene.",
    }


def _probe_dynamic_object_stable(spec: dict[str, Any], *, object_id: str | None) -> dict[str, Any]:
    mujoco, model, data = _load_mujoco_model(spec)
    targets = _dynamic_probe_objects(spec, object_id=object_id)
    if not targets:
        return {
            "probe": "dynamic_object_stable",
            "passed": False,
            "issue": "No matching dynamic object was found.",
        }
    initial = _body_positions(mujoco, model, data, targets)
    min_z = math.inf
    for _ in range(260):
        mujoco.mj_step(model, data)
        if not _data_finite(data):
            return {"probe": "dynamic_object_stable", "passed": False, "issue": "simulation became non-finite"}
        for pos in _body_positions(mujoco, model, data, targets).values():
            min_z = min(min_z, pos[2])
    final = _body_positions(mujoco, model, data, targets)
    max_displacement = _max_displacement(initial, final)
    passed = max_displacement <= 1.5 and min_z > -1.0
    return {
        "probe": "dynamic_object_stable",
        "passed": passed,
        "body_ids": [item.id for item in targets],
        "max_displacement": round(max_displacement, 4),
        "min_z": round(min_z, 4),
        "issue": "" if passed else "A dynamic object moved too far or fell during passive simulation.",
    }


def _probe_pushable_moves(spec: dict[str, Any], *, object_id: str | None) -> dict[str, Any]:
    mujoco, model, data = _load_mujoco_model(spec)
    targets = _dynamic_probe_objects(spec, object_id=object_id, pushable_only=True)
    if not targets:
        return {
            "probe": "pushable_moves",
            "passed": False,
            "issue": "No matching pushable dynamic object was found.",
        }
    target = targets[0]
    body_id = _body_id(mujoco, model, target.id)
    if body_id < 0:
        return {"probe": "pushable_moves", "passed": False, "issue": f"MuJoCo body {target.id!r} was not found."}
    mujoco.mj_forward(model, data)
    initial = _body_position(data, body_id)
    for _ in range(100):
        data.xfrc_applied[body_id, 0] = 30.0
        mujoco.mj_step(model, data)
        data.xfrc_applied[body_id, 0] = 0.0
        if not _data_finite(data):
            return {"probe": "pushable_moves", "passed": False, "issue": "simulation became non-finite"}
    for _ in range(30):
        mujoco.mj_step(model, data)
    final = _body_position(data, body_id)
    displacement_xy = math.dist(initial[:2], final[:2])
    passed = displacement_xy >= 0.15
    return {
        "probe": "pushable_moves",
        "passed": passed,
        "body_id": target.id,
        "displacement_xy": round(displacement_xy, 4),
        "initial_position": [round(value, 4) for value in initial],
        "final_position": [round(value, 4) for value in final],
        "issue": "" if passed else "The selected pushable object did not move under a deterministic horizontal force.",
    }


def _load_mujoco_model(spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    try:
        import mujoco
    except Exception as exc:  # pragma: no cover - depends on local install
        raise EnvVerificationError(f"MuJoCo is unavailable: {exc}") from exc
    model = mujoco.MjModel.from_xml_string(compile_spec_to_mjcf(spec))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _dynamic_probe_objects(
    spec: dict[str, Any],
    *,
    object_id: str | None = None,
    pushable_only: bool = False,
) -> list[SceneObject3D]:
    objects = [
        obj
        for obj in scene_objects(spec)
        if obj.body_type == "dynamic" and (object_id is None or obj.id == object_id)
    ]
    if pushable_only:
        objects = [
            obj
            for obj in objects
            if obj.semantic_type in PUSHABLE_TYPES or "pushable" in obj.tags
        ]
    return objects


def _body_positions(mujoco: Any, model: Any, data: Any, objects: list[SceneObject3D]) -> dict[str, tuple[float, float, float]]:
    positions = {}
    for obj in objects:
        body_id = _body_id(mujoco, model, obj.id)
        if body_id >= 0:
            positions[obj.id] = _body_position(data, body_id)
    return positions


def _body_id(mujoco: Any, model: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _body_position(data: Any, body_id: int) -> tuple[float, float, float]:
    return (
        float(data.xpos[body_id, 0]),
        float(data.xpos[body_id, 1]),
        float(data.xpos[body_id, 2]),
    )


def _max_body_speed(mujoco: Any, model: Any, data: Any, objects: list[SceneObject3D]) -> float:
    max_speed = 0.0
    for obj in objects:
        body_id = _body_id(mujoco, model, obj.id)
        if body_id < 0:
            continue
        velocity = data.cvel[body_id]
        max_speed = max(max_speed, math.sqrt(sum(float(value) ** 2 for value in velocity[:3])))
    return max_speed


def _data_finite(data: Any) -> bool:
    for array in (data.qpos, data.qvel, data.xpos):
        for value in array.flat:
            if not math.isfinite(float(value)):
                return False
    return True


def _max_displacement(
    initial: dict[str, tuple[float, float, float]],
    final: dict[str, tuple[float, float, float]],
) -> float:
    distances = [math.dist(start, final[obj_id]) for obj_id, start in initial.items() if obj_id in final]
    return max(distances, default=0.0)


def _scene_object_mapping(obj: SceneObject3D) -> dict[str, Any]:
    return {
        "id": obj.id,
        "semantic_type": obj.semantic_type,
        "shape": obj.shape,
        "body_type": obj.body_type,
        "position": list(obj.position),
        "size": list(obj.size),
        "yaw": obj.yaw,
        "metadata": obj.metadata,
    }


def _best_endpoint_surface_match(
    endpoint: tuple[float, float, float],
    surfaces: list[SceneObject3D],
) -> dict[str, Any]:
    candidates = [_endpoint_surface_measurement(endpoint, surface) for surface in surfaces]
    return min(
        candidates,
        key=lambda item: math.hypot(float(item["horizontal_gap"]), float(item["vertical_gap"])),
    )


def _endpoint_surface_measurement(
    endpoint: tuple[float, float, float],
    surface: SceneObject3D,
) -> dict[str, Any]:
    dx = endpoint[0] - surface.position[0]
    dy = endpoint[1] - surface.position[1]
    c_yaw = math.cos(surface.yaw)
    s_yaw = math.sin(surface.yaw)
    local_x = dx * c_yaw + dy * s_yaw
    local_y = -dx * s_yaw + dy * c_yaw
    outside_x = max(0.0, abs(local_x) - surface.size[0] / 2.0)
    outside_y = max(0.0, abs(local_y) - surface.size[1] / 2.0)
    horizontal_gap = math.hypot(outside_x, outside_y)
    if surface.shape == "ramp":
        geometry = ramp_geometry_from_object(_scene_object_mapping(surface))
        surface_z = ramp_surface_height(geometry, endpoint[0], endpoint[1])
        if surface_z is None:
            surface_z = min(
                (geometry.low_end[2], geometry.high_end[2]),
                key=lambda value: abs(endpoint[2] - value),
            )
    else:
        surface_z = surface.position[2] + surface.size[2] / 2.0
    return {
        "id": surface.id,
        "horizontal_gap": round(horizontal_gap, 4),
        "vertical_gap": round(abs(endpoint[2] - surface_z), 4),
        "surface_z": round(surface_z, 4),
    }


def _endpoint_match_passes(match: dict[str, Any], check: dict[str, Any]) -> bool:
    return (
        float(match["horizontal_gap"]) <= float(check["max_horizontal_gap"])
        and float(match["vertical_gap"]) <= float(check["max_vertical_gap"])
    )


def _object_bounds(obj: dict[str, Any]) -> dict[str, float]:
    x, y, z = [float(value) for value in obj["position"]]
    sx, sy, sz = [float(value) for value in obj["size"]]
    shape = str(obj["shape"])
    if shape == "sphere":
        radius = sx / 2.0
        return {
            "x1": x - radius,
            "x2": x + radius,
            "y1": y - radius,
            "y2": y + radius,
            "z1": z - radius,
            "z2": z + radius,
        }
    if shape in {"cylinder", "capsule"}:
        radius = sx / 2.0
        return {
            "x1": x - radius,
            "x2": x + radius,
            "y1": y - radius,
            "y2": y + radius,
            "z1": z - sz / 2.0,
            "z2": z + sz / 2.0,
        }
    yaw = float(obj.get("yaw") or 0.0)
    half_x = sx / 2.0
    half_y = sy / 2.0
    corners = [
        _rotate_xy(-half_x, -half_y, yaw),
        _rotate_xy(-half_x, half_y, yaw),
        _rotate_xy(half_x, -half_y, yaw),
        _rotate_xy(half_x, half_y, yaw),
    ]
    xs = [x + corner[0] for corner in corners]
    ys = [y + corner[1] for corner in corners]
    if shape == "ramp":
        return ramp_bounds(ramp_geometry_from_object(obj))
    else:
        z1 = z - sz / 2.0
        z2 = z + sz / 2.0
    return {"x1": min(xs), "x2": max(xs), "y1": min(ys), "y2": max(ys), "z1": z1, "z2": z2}


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return x * c - y * s, x * s + y * c


def _select_objects(objects: list[SceneObject3D], selector: dict[str, Any]) -> list[SceneObject3D]:
    return [obj for obj in objects if _matches_selector(obj, selector)]


def _matches_selector(obj: SceneObject3D, selector: dict[str, Any]) -> bool:
    if selector.get("id") is not None and not _value_matches(obj.id, selector["id"], frozenset()):
        return False
    if selector.get("semantic_type") is not None and not _value_matches(obj.semantic_type, selector["semantic_type"], obj.tags):
        return False
    if selector.get("body_type") is not None and not _value_matches(obj.body_type, selector["body_type"], obj.tags):
        return False
    if selector.get("shape") is not None and not _value_matches(obj.shape, selector["shape"], obj.tags):
        return False
    if selector.get("tag") is not None and not _value_matches("", selector["tag"], obj.tags):
        return False
    return True


def _value_matches(primary: str, expected: Any, tags: frozenset[str]) -> bool:
    values = expected if isinstance(expected, list) else [expected]
    wanted = {str(value).lower() for value in values}
    return primary.lower() in wanted or bool(tags.intersection(wanted))


def _relation_holds(subject: SceneObject3D, target: SceneObject3D, check: dict[str, Any]) -> bool:
    margin = float(check.get("margin") or 0.0)
    relation = check["relation"]
    sb = subject.bounds
    tb = target.bounds
    if relation == "left_of":
        return sb["x2"] <= tb["x1"] + margin
    if relation == "right_of":
        return sb["x1"] >= tb["x2"] - margin
    if relation == "in_front_of":
        return sb["y1"] >= tb["y2"] - margin
    if relation == "behind":
        return sb["y2"] <= tb["y1"] + margin
    if relation == "above":
        return sb["z1"] >= tb["z2"] - margin
    if relation == "below":
        return sb["z2"] <= tb["z1"] + margin
    if relation == "near":
        return _distance(_center(subject), _center(target)) <= float(check["max_distance"])
    if relation == "far_from":
        return _distance(_center(subject), _center(target)) >= float(check["min_distance"])
    if relation == "inside":
        return volume_contains(target, subject, margin=margin)
    if relation == "on_surface":
        if target.shape == "ramp":
            surface_z = ramp_surface_height(
                ramp_geometry_from_object(_scene_object_mapping(target)),
                subject.position[0],
                subject.position[1],
            )
            if surface_z is None:
                return False
        else:
            surface_z = tb["z2"]
        vertical_gap = abs(sb["z1"] - surface_z)
        return footprints_overlap(subject, target) and vertical_gap <= max(0.18, margin)
    raise AssertionError("unreachable relation")


def _is_between(subject: SceneObject3D, a: SceneObject3D, b: SceneObject3D, *, margin: float) -> bool:
    center = _center(subject)
    ac = _center(a)
    bc = _center(b)
    ab = (bc[0] - ac[0], bc[1] - ac[1], bc[2] - ac[2])
    ap = (center[0] - ac[0], center[1] - ac[1], center[2] - ac[2])
    length_sq = sum(value * value for value in ab)
    if length_sq <= 1e-12:
        return False
    t = sum(ap[index] * ab[index] for index in range(3)) / length_sq
    return -margin <= t <= 1.0 + margin


def _best_support(
    subject: SceneObject3D,
    surfaces: list[SceneObject3D],
) -> tuple[SceneObject3D, float, float] | None:
    candidates = []
    for surface in surfaces:
        if surface.id == subject.id:
            continue
        overlap = _xy_overlap_area(subject, surface)
        if overlap <= 0.0:
            continue
        if surface.shape == "ramp":
            surface_z = ramp_surface_height(
                ramp_geometry_from_object(_scene_object_mapping(surface)),
                subject.position[0],
                subject.position[1],
            )
            if surface_z is None:
                continue
            vertical_gap = subject.bounds["z1"] - surface_z
        else:
            vertical_gap = subject.bounds["z1"] - surface.bounds["z2"]
        candidates.append((abs(vertical_gap), surface, overlap, vertical_gap))
    if not candidates:
        return None
    _, surface, overlap, vertical_gap = min(candidates, key=lambda item: item[0])
    return surface, overlap, vertical_gap


def _xy_overlap_area(a: SceneObject3D, b: SceneObject3D) -> float:
    if not footprints_overlap(a, b):
        return 0.0
    x_overlap = min(a.bounds["x2"], b.bounds["x2"]) - max(a.bounds["x1"], b.bounds["x1"])
    y_overlap = min(a.bounds["y2"], b.bounds["y2"]) - max(a.bounds["y1"], b.bounds["y1"])
    return max(0.0, x_overlap) * max(0.0, y_overlap)


def _center(obj: SceneObject3D) -> tuple[float, float, float]:
    return (
        (obj.bounds["x1"] + obj.bounds["x2"]) / 2.0,
        (obj.bounds["y1"] + obj.bounds["y2"]) / 2.0,
        (obj.bounds["z1"] + obj.bounds["z2"]) / 2.0,
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def _check_result(
    check: dict[str, Any],
    *,
    passed: bool,
    message: str,
    metrics: dict[str, Any] | None = None,
    repair_hints: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": check.get("id"),
        "type": check.get("type"),
        "severity": check.get("severity", "critical"),
        "description": check.get("description") or "",
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "message": message,
        "metrics": metrics or {},
        "repair_hints": repair_hints or [],
    }


def _report_next_action(critical_failures: int, advisory_failures: int) -> str:
    if critical_failures:
        return "Repair critical environment check failures, then rerun run_env_verification."
    if advisory_failures:
        return "Review advisory issues. They do not block finalization."
    return "Environment verification passed. Continue validation and finalization."


def _count_repair_hint(check: dict[str, Any], count: int) -> str:
    selector = check.get("selector") or {}
    semantic = selector.get("semantic_type") or selector.get("tag") or selector.get("id") or "matching object"
    if "min" in check and count < check["min"]:
        return f"Add or retag {semantic} so the environment contains the required object count."
    if "exact" in check and count > check["exact"]:
        return f"Remove duplicate {semantic} objects or tighten the selector."
    return f"Adjust {semantic} count or selector to match the prompt-derived requirement."


def _spatial_repair_hint(relation: str) -> str:
    if relation == "left_of":
        return "Move the subject to lower world x than the target."
    if relation == "right_of":
        return "Move the subject to higher world x than the target."
    if relation == "in_front_of":
        return "Move the subject to higher world y than the target."
    if relation == "behind":
        return "Move the subject to lower world y than the target."
    if relation == "above":
        return "Raise the subject above the target."
    if relation == "below":
        return "Lower the subject below the target."
    if relation == "between":
        return "Move the subject between the two target objects."
    if relation == "near":
        return "Move the subject closer to the target."
    if relation == "far_from":
        return "Increase the distance between the subject and target."
    if relation == "inside":
        return "Move or resize the subject so it fits inside the target volume."
    return "Place the subject on or immediately above the target surface."


def _physics_repair_hint(probe: str, probe_result: dict[str, Any]) -> str:
    if probe_result.get("issue"):
        return str(probe_result["issue"])
    if probe == "pushable_moves":
        return "Ensure the selected object is dynamic, pushable, and not trapped by static geometry."
    if probe == "dynamic_object_stable":
        return "Place dynamic objects on support surfaces so they do not fall or drift during passive simulation."
    return "Fix unstable MuJoCo geometry or unsupported dynamic objects and rerun the probe."


def _report_is_stale(
    report: dict[str, Any],
    *,
    draft_hash: str | None,
    operation_count: int | None,
) -> bool:
    if draft_hash:
        report_hash = str(report.get("draft_hash") or "")
        if report_hash and report_hash != draft_hash:
            return True
    if operation_count is not None:
        report_operation_count = _optional_int(report.get("operation_count"))
        if report_operation_count is not None and report_operation_count != operation_count:
            return True
    return False


def _normalize_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    if not cleaned:
        raise EnvVerificationError("check id cannot be blank")
    return cleaned[:80]


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise EnvVerificationError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise EnvVerificationError(f"{name} must be non-negative")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise EnvVerificationError(f"{name} must be positive")
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tuple3(values: Iterable[Any]) -> tuple[float, float, float]:
    x, y, z = list(values)
    return float(x), float(y), float(z)


def _spec_json(spec: dict[str, Any] | EnvSpec3D) -> dict[str, Any]:
    return env_spec_to_dict(spec) if isinstance(spec, EnvSpec3D) else spec


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
