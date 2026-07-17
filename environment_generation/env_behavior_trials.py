"""Prompt-derived, code-scored behavior trials for generated 3D scenes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env_tasks import (
    TaskDefinitionError,
    describe_assertion_condition,
    normalize_assertion_group,
)
from .env_verification import (
    SUPPORTED_SPATIAL_RELATIONS,
    normalize_selector,
    scene_objects,
    select_scene_objects,
    spec_hash,
)
from .schema import EnvSpec3D, env_spec_to_dict, parse_env_spec_3d


ENV_BEHAVIOR_TRIAL_PLAN_FILENAME = "env_behavior_trials_plan.json"
ENV_BEHAVIOR_TRIAL_REPORT_FILENAME = "env_behavior_trials_report.json"
ENV_BEHAVIOR_TRIALS_DIRNAME = "behavior_trials"
ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION = "2.0"
BEHAVIOR_CONTROLLER_VERSION = "controller_v3"

SUPPORTED_EXPECTED_OUTCOMES = {"should_succeed", "should_not_succeed"}
SUPPORTED_BEHAVIOR_PLAN_DECISIONS = {"prompt_specific", "preserved", "fallback"}
CANONICAL_BEHAVIOR_RESULT_STATUSES = {
    "passed",
    "failed",
    "inconclusive",
    "invalid_setup",
    "error",
}
SUPPORTED_BEHAVIOR_CHECK_TYPES = {
    "agent_displacement",
    "agent_relation",
    "agent_height_gain",
    "zone_entry",
    "object_displacement",
    "contact_count",
    "jump_count",
    "mechanism_state",
    "terminal_event",
    "attempt_reset",
}
SUPPORTED_TERMINAL_EVENTS = {"hazard", "out_of_bounds", "goal"}
SUPPORTED_RESET_REASONS = {
    "any",
    *SUPPORTED_TERMINAL_EVENTS,
    "attempt_budget",
    "step_budget",
    "manual",
}
SUPPORTED_SEVERITIES = {"critical", "advisory"}
SUPPORTED_CHECK_TIMES = {"ever", "final"}
SUPPORTED_RELATIONS = SUPPORTED_SPATIAL_RELATIONS - {"between", "inside"}
MAX_TRIALS = 2
DEFAULT_MAX_STEPS = 2400
MAX_STEPS = 3600
DEFAULT_MAX_RESETS = 1
MAX_RESETS = 2
NEGATIVE_SEARCH_MIN_ACTIVE_STEPS = 60
NEGATIVE_SEARCH_MAX_ACTIVE_STEPS = 240
NEGATIVE_SEARCH_BUDGET_FRACTION = 0.10


class EnvBehaviorTrialError(ValueError):
    """Raised when a behavior-trial plan is malformed."""


def normalize_behavior_trial_plan(
    *,
    env_id: str,
    prompt: str,
    trials: Any,
    operation_count: int,
    draft_spec: dict[str, Any] | EnvSpec3D,
    fallback: bool = False,
    decision: str | None = None,
    decision_reason: str = "",
    source_turn_id: str = "",
    intent_summary: str = "",
) -> dict[str, Any]:
    spec = draft_spec if isinstance(draft_spec, EnvSpec3D) else parse_env_spec_3d(draft_spec)
    if not _agent_objects(spec):
        raise EnvBehaviorTrialError("behavior trials require an authored agent")
    if not isinstance(trials, list) or not trials:
        raise EnvBehaviorTrialError("trials must be a non-empty list")
    if len(trials) > MAX_TRIALS:
        raise EnvBehaviorTrialError(f"trials cannot contain more than {MAX_TRIALS} items")
    resolved_decision = str(decision or ("fallback" if fallback else "prompt_specific")).strip()
    if resolved_decision not in SUPPORTED_BEHAVIOR_PLAN_DECISIONS:
        raise EnvBehaviorTrialError(f"unsupported behavior-plan decision {resolved_decision!r}")
    normalized = [
        _normalize_trial(index, trial, spec=spec)
        for index, trial in enumerate(trials)
    ]
    mechanisms_by_id = {mechanism.id: mechanism for mechanism in spec.mechanisms}
    for trial in normalized:
        groups = [trial.get("objective") or {}, trial.get("constraints") or {}]
        for group in groups:
            for check in group.get("checks") or []:
                predicate = check.get("predicate") or {}
                if predicate.get("type") == "mechanism_state":
                    mechanism = mechanisms_by_id.get(predicate.get("mechanism_id"))
                    if mechanism is None:
                        raise EnvBehaviorTrialError(
                            "mechanism_state references unknown mechanism "
                            f"{predicate.get('mechanism_id')!r}"
                        )
                    check["selector"] = {"id": mechanism.trigger_id}
    trial_ids = [trial["id"] for trial in normalized]
    if len(trial_ids) != len(set(trial_ids)):
        raise EnvBehaviorTrialError("trial ids must be unique")
    draft_json = env_spec_to_dict(spec)
    return {
        "schema_version": ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION,
        "controller_version": BEHAVIOR_CONTROLLER_VERSION,
        "env_id": env_id,
        "prompt": " ".join(str(prompt).split())[:12000],
        "operation_count": int(operation_count),
        "draft_hash": spec_hash(draft_json),
        "created_at": _now(),
        "fallback": bool(fallback),
        "decision": resolved_decision,
        "decision_reason": _single_line(decision_reason)[:2000],
        "source_turn_id": str(source_turn_id or "").strip()[:200],
        "source_prompt_hash": behavior_prompt_hash(prompt),
        "intent_summary": _single_line(intent_summary)[:4000],
        "trials": normalized,
    }


def normalize_behavior_trial_for_runtime(
    trial: dict[str, Any],
    *,
    spec: EnvSpec3D,
) -> dict[str, Any]:
    """Upgrade one persisted or directly supplied trial before execution."""

    groups = [trial.get("objective") or {}, trial.get("constraints") or {}]
    checks = [
        check
        for group in groups
        for check in group.get("checks") or group.get("conditions") or []
        if isinstance(check, dict)
    ]
    if checks and all(isinstance(check.get("predicate"), dict) for check in checks):
        _normalize_trial(0, trial, spec=spec)
        return trial
    normalized = _normalize_trial(0, trial, spec=spec)
    return {**trial, **normalized}


def fallback_locomotion_plan(
    *,
    env_id: str,
    prompt: str,
    operation_count: int,
    draft_spec: dict[str, Any] | EnvSpec3D,
    source_turn_id: str = "",
    decision_reason: str = "",
) -> dict[str, Any] | None:
    spec = draft_spec if isinstance(draft_spec, EnvSpec3D) else parse_env_spec_3d(draft_spec)
    if not _agent_objects(spec):
        return None
    if spec.game:
        checks: list[dict[str, Any]] = [
            {
                "id": "enter_goal",
                "type": "zone_entry",
                "selector": {"id": spec.game.goal_id},
                "min_count": 1,
                "description": "The robot enters the charging-pad goal zone.",
            }
        ]
        ordered: list[str] = []
        instruction = "Navigate through the courtyard and enter the charging-pad goal zone."
        if spec.mechanisms:
            mechanism = spec.mechanisms[0]
            checks.insert(
                0,
                {
                    "id": "open_gate",
                    "type": "mechanism_state",
                    "mechanism_id": mechanism.id,
                    "state": "open",
                    "min_progress": 0.9,
                    "description": "Activate the floor switch and open its linked gate.",
                },
            )
            ordered = ["open_gate", "enter_goal"]
            instruction = "Activate the courtyard floor switch, pass through the opened gate, and enter the goal zone."
        return normalize_behavior_trial_plan(
            env_id=env_id,
            prompt=prompt,
            operation_count=operation_count,
            draft_spec=spec,
            fallback=True,
            decision="fallback",
            decision_reason=decision_reason,
            source_turn_id=source_turn_id,
            intent_summary="Use the standard courtyard route to demonstrate that the authored game remains playable.",
            trials=[
                {
                    "id": "reach_courtyard_goal",
                    "instruction": instruction,
                    "expected_outcome": "should_succeed",
                    "severity": "advisory",
                    "objective": {"mode": "all", "checks": checks, "ordered_check_ids": ordered},
                }
            ],
        )
    return normalize_behavior_trial_plan(
        env_id=env_id,
        prompt=prompt,
        operation_count=operation_count,
        draft_spec=spec,
        fallback=True,
        decision="fallback",
        decision_reason=decision_reason,
        source_turn_id=source_turn_id,
        intent_summary="Demonstrate basic locomotion and stability because no concrete affordance was requested.",
        trials=[
            {
                "id": "basic_locomotion",
                "instruction": (
                    "Move at least one meter away from the authored spawn while remaining "
                    "stable in the environment. Stop once that movement is demonstrated."
                ),
                "expected_outcome": "should_succeed",
                "severity": "advisory",
                "objective": {
                    "mode": "all",
                    "checks": [
                        {
                            "id": "agent_moves_one_meter",
                            "type": "agent_displacement",
                            "min_distance": 1.0,
                            "space": "xy",
                        },
                        {
                            "id": "agent_finishes_supported",
                            "type": "agent_relation",
                            "target": {"body_type": "static"},
                            "relation": "on_surface",
                            "when": "final",
                            "description": "Agent finishes supported by an authored static surface.",
                        },
                    ],
                },
            }
        ],
    )


def behavior_plan_hash(plan: dict[str, Any]) -> str:
    relevant = {
        "schema_version": plan.get("schema_version"),
        "controller_version": plan.get("controller_version"),
        "env_id": plan.get("env_id"),
        "draft_hash": plan.get("draft_hash"),
        "decision": plan.get("decision"),
        "source_turn_id": plan.get("source_turn_id"),
        "source_prompt_hash": plan.get("source_prompt_hash"),
        "intent_summary": plan.get("intent_summary"),
        "trials": plan.get("trials") or [],
    }
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def behavior_prompt_hash(prompt: str) -> str:
    normalized = _single_line(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def behavior_plan_decision_issues(
    plan: dict[str, Any] | None,
    *,
    current_spec: dict[str, Any] | EnvSpec3D,
    prompt: str,
    source_turn_id: str,
) -> list[str]:
    spec = current_spec if isinstance(current_spec, EnvSpec3D) else parse_env_spec_3d(current_spec)
    if not _agent_objects(spec):
        return []
    if not isinstance(plan, dict):
        return ["No current-turn agent-test decision was recorded."]
    issues: list[str] = []
    if plan.get("draft_hash") != spec_hash(spec):
        issues.append("The agent-test plan does not match the current scene draft.")
    if plan.get("controller_version") != BEHAVIOR_CONTROLLER_VERSION or plan.get("migration_issues"):
        issues.append("The agent-test plan does not use the current controller contract.")
    decision = str(plan.get("decision") or "")
    if decision not in SUPPORTED_BEHAVIOR_PLAN_DECISIONS:
        issues.append("The current turn has no explicit agent-test decision.")
    if plan.get("source_prompt_hash") != behavior_prompt_hash(prompt):
        issues.append("The agent-test decision was made for a different user request.")
    if not source_turn_id or plan.get("source_turn_id") != source_turn_id:
        issues.append("The agent-test decision was not made during the current Studio turn.")
    if not str(plan.get("decision_reason") or "").strip():
        issues.append("The agent-test decision is missing its rationale.")
    if decision == "prompt_specific" and not str(plan.get("intent_summary") or "").strip():
        issues.append("The prompt-specific agent-test plan is missing an intent summary.")
    if not isinstance(plan.get("trials"), list) or not plan.get("trials"):
        issues.append("The agent-test decision contains no runnable tests.")
    return issues


def behavior_trial_summary(
    scene_dir: Path,
    *,
    current_spec: dict[str, Any] | EnvSpec3D | None = None,
    operation_count: int | None = None,
) -> dict[str, Any]:
    spec = _optional_spec(current_spec)
    plan = load_behavior_trial_plan(scene_dir, current_spec=spec)
    report = load_behavior_trial_report(scene_dir, plan=plan)
    active_runs = active_behavior_runs(scene_dir)
    active = active_runs[-1] if active_runs else None
    history = behavior_trial_history(scene_dir)
    if spec is not None and not _agent_objects(spec):
        return {
            "status": "not_applicable",
            "label": "Behavior trials: not applicable",
            "has_plan": False,
            "has_report": False,
            "trial_count": 0,
            "active_run": active,
            "active_runs": active_runs,
            "history_count": len(history),
            "message": "This scene has no authored controllable agent.",
        }
    if active_runs:
        return {
            "status": str(active.get("status") or "running"),
            "label": "Behavior trials: running",
            "has_plan": bool(plan),
            "has_report": bool(report),
            "trial_count": len((plan or {}).get("trials") or []),
            "active_run": active,
            "active_runs": active_runs,
            "active_run_count": len(active_runs),
            "history_count": len(history),
        }
    if not plan:
        return {
            "status": "missing",
            "label": "Behavior trials: not defined",
            "has_plan": False,
            "has_report": False,
            "trial_count": 0,
            "history_count": len(history),
        }
    current_hash = spec_hash(spec) if spec is not None else None
    plan_stale = bool(
        (current_hash and plan.get("draft_hash") != current_hash)
        or plan.get("controller_version") != BEHAVIOR_CONTROLLER_VERSION
        or plan.get("migration_issues")
    )
    dismissed = bool(plan.get("dismissed"))
    if dismissed and not report:
        return {
            "status": "dismissed",
            "label": "Behavior trials: dismissed",
            "has_plan": True,
            "has_report": False,
            "trial_count": len(plan.get("trials") or []),
            "fallback": bool(plan.get("fallback")),
            "decision": plan.get("decision"),
            "migration_issues": list(plan.get("migration_issues") or []),
            "history_count": len(history),
        }
    if plan_stale:
        return {
            "status": "stale",
            "label": "Behavior trials: needs regeneration",
            "has_plan": True,
            "has_report": bool(report),
            "stale": True,
            "trial_count": len(plan.get("trials") or []),
            "fallback": bool(plan.get("fallback")),
            "decision": plan.get("decision"),
            "history_count": len(history),
        }
    if not report:
        return {
            "status": "ready_to_run",
            "label": "Behavior trials: ready",
            "has_plan": True,
            "has_report": False,
            "trial_count": len(plan.get("trials") or []),
            "fallback": bool(plan.get("fallback")),
            "decision": plan.get("decision"),
            "history_count": len(history),
        }
    report_stale = (
        report.get("env_spec_hash") != current_hash
        or report.get("behavior_plan_hash") != behavior_plan_hash(plan)
        or report.get("controller_version") != BEHAVIOR_CONTROLLER_VERSION
    )
    if report_stale:
        status = "stale"
        label = "Behavior trials: stale"
    else:
        status = str(report.get("status") or "error")
        label = {
            "passed": "Behavior trials: passed",
            "needs_attention": "Behavior trials: needs attention",
            "failed": "Behavior trials: failed",
            "error": "Behavior trials: error",
            "partial": "Behavior trials: partially run",
        }.get(status, f"Behavior trials: {status.replace('_', ' ')}")
    return {
        "status": status,
        "label": label,
        "has_plan": True,
        "has_report": True,
        "stale": report_stale,
        "trial_count": len(plan.get("trials") or []),
        "fallback": bool(plan.get("fallback")),
        "decision": plan.get("decision"),
        "summary": report.get("summary") or {},
        "operation_count": operation_count,
        "history_count": len(history),
    }


def build_behavior_trial_report(
    *,
    env_id: str,
    plan: dict[str, Any],
    env_spec_hash: str,
    run_id: str,
    results: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    trials_by_id = {
        str(trial.get("id")): trial
        for trial in plan.get("trials") or []
        if isinstance(trial, dict) and trial.get("id")
    }
    known_trial_ids = set(trials_by_id)
    normalized_results = [
        normalize_behavior_trial_result(
            result,
            trial=trials_by_id.get(str(result.get("trial_id") or "")),
        )
        for result in results
        if isinstance(result, dict)
    ]
    result_trial_ids = {
        str(result.get("trial_id"))
        for result in normalized_results
        if result.get("trial_id")
    }
    counts = _behavior_result_counts(
        results=normalized_results,
        total=len(known_trial_ids),
        result_trial_ids=result_trial_ids,
    )
    status = _behavior_report_status(counts)
    return {
        "schema_version": ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION,
        "controller_version": BEHAVIOR_CONTROLLER_VERSION,
        "env_id": env_id,
        "run_id": run_id,
        "created_at": _now(),
        "status": status,
        "passed": status == "passed",
        "blocking": False,
        "env_spec_hash": env_spec_hash,
        "behavior_plan_hash": behavior_plan_hash(plan),
        "model": model or "default",
        "summary": {"total": len(known_trial_ids), **counts},
        "results": normalized_results,
        "next_action": _behavior_report_next_action(counts),
    }


def normalize_behavior_trial_result(
    result: dict[str, Any],
    *,
    trial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one result using the canonical passed/failed vocabulary."""

    normalized = dict(result)
    source_trial = trial or {}
    for group_name in ("objective", "constraints"):
        group = normalized.get(group_name)
        if isinstance(group, dict):
            normalized[group_name] = _normalize_result_group_descriptions(
                group,
                source_trial.get(group_name) if isinstance(source_trial.get(group_name), dict) else {},
            )
    expected = str(
        normalized.get("expected_outcome")
        or (trial or {}).get("expected_outcome")
        or "should_succeed"
    )
    status = str(normalized.get("status") or "error")
    normalized["expected_outcome"] = expected
    compatible_statuses = CANONICAL_BEHAVIOR_RESULT_STATUSES | {
        "demonstrated",
        "not_observed",
    }
    if status not in compatible_statuses:
        status = "error"

    if expected == "should_not_succeed" and status not in {"error", "invalid_setup"}:
        objective = normalized.get("objective") if isinstance(normalized.get("objective"), dict) else {}
        if objective.get("satisfied") is True or status == "failed":
            status = "failed"
        else:
            search = normalized.get("search_evidence")
            if not isinstance(search, dict) or not isinstance(search.get("valid"), bool):
                search = negative_search_evidence(
                    trial=trial or {},
                    actions=normalized.get("actions") or [],
                    objective=objective,
                    constraints=(
                        normalized.get("constraints")
                        if isinstance(normalized.get("constraints"), dict)
                        else None
                    ),
                    termination_reason=str(normalized.get("termination_reason") or ""),
                )
            normalized["search_evidence"] = search
            status = "passed" if search.get("valid") is True else "inconclusive"
    elif status == "demonstrated":
        status = "passed"
    elif status == "not_observed":
        status = "inconclusive"

    normalized["status"] = status
    normalized["passed"] = status == "passed"
    normalized["non_failure"] = status == "passed"
    if status == "passed":
        normalized["repair_hints"] = []
    elif expected == "should_not_succeed" and status == "inconclusive" and not normalized.get("repair_hints"):
        normalized["repair_hints"] = [negative_search_repair_hint(normalized.get("search_evidence") or {})]
    elif status == "inconclusive":
        failed = [
            check
            for check in (normalized.get("objective") or {}).get("checks") or []
            if check.get("passed") is False
        ]
        hints = normalized.get("repair_hints") or []
        generated_hint = bool(hints) and str(hints[0]).startswith(
            "Review or repair the unmet objective checks:"
        )
        if failed and (not hints or generated_hint):
            names = [describe_assertion_condition(check).rstrip(". ") for check in failed]
            normalized["repair_hints"] = [
                "Review or repair the unmet objective checks: " + ", ".join(names[:5]) + "."
            ]
    return normalized


def _normalize_result_group_descriptions(
    group: dict[str, Any],
    source_group: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(group)
    source_by_id = {
        str(check.get("id")): check
        for check in source_group.get("checks") or source_group.get("conditions") or []
        if isinstance(check, dict) and check.get("id")
    }
    checks = []
    for raw_check in group.get("checks") or group.get("conditions") or []:
        if not isinstance(raw_check, dict):
            continue
        check = dict(raw_check)
        source = source_by_id.get(str(check.get("id") or ""), {})
        check["description"] = describe_assertion_condition(
            check,
            fallback=str(source.get("description") or ""),
        )
        checks.append(check)
    normalized["checks"] = checks
    return normalized


def normalize_behavior_trial_report(
    report: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upgrade persisted reports without rewriting historical evidence."""

    normalized = dict(report)
    trials_by_id = {
        str(trial.get("id")): trial
        for trial in (plan or {}).get("trials") or []
        if isinstance(trial, dict) and trial.get("id")
    }
    results = [
        normalize_behavior_trial_result(
            result,
            trial=trials_by_id.get(str(result.get("trial_id") or "")),
        )
        for result in report.get("results") or []
        if isinstance(result, dict)
    ]
    result_trial_ids = {
        str(result.get("trial_id"))
        for result in results
        if result.get("trial_id")
    }
    raw_total = _coerce_bounded_runtime_int(
        (report.get("summary") or {}).get("total"),
        0,
        0,
        MAX_TRIALS,
    )
    total = max(len(trials_by_id), len(result_trial_ids), raw_total)
    counts = _behavior_result_counts(
        results=results,
        total=total,
        result_trial_ids=result_trial_ids,
    )
    status = _behavior_report_status(counts)
    normalized.update(
        {
            "status": status,
            "passed": status == "passed",
            "summary": {"total": total, **counts},
            "results": results,
            "next_action": _behavior_report_next_action(counts),
        }
    )
    return normalized


def negative_search_evidence(
    *,
    trial: dict[str, Any],
    actions: list[dict[str, Any]],
    objective: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    termination_reason: str,
) -> dict[str, Any]:
    """Measure whether a bounded negative trial performed a meaningful search."""

    max_steps = _coerce_bounded_runtime_int(trial.get("max_steps"), DEFAULT_MAX_STEPS, 60, MAX_STEPS)
    max_resets = _coerce_bounded_runtime_int(trial.get("max_resets"), DEFAULT_MAX_RESETS, 0, MAX_RESETS)
    max_total_steps = _coerce_bounded_runtime_int(
        trial.get("max_total_steps"),
        max_steps * (max_resets + 1),
        60,
        MAX_STEPS * (MAX_RESETS + 1),
    )
    required_active_steps = min(
        max_total_steps,
        max(
            NEGATIVE_SEARCH_MIN_ACTIVE_STEPS,
            min(
                NEGATIVE_SEARCH_MAX_ACTIVE_STEPS,
                int(math.ceil(max_total_steps * NEGATIVE_SEARCH_BUDGET_FRACTION)),
            ),
        ),
    )
    active_steps = 0
    active_segments = 0
    for action in actions:
        if not isinstance(action, dict) or action.get("action") != "controller":
            continue
        active_control = (
            abs(_safe_runtime_float(action.get("forward"))) > 0.05
            or abs(_safe_runtime_float(action.get("right"))) > 0.05
            or bool(action.get("jump"))
        )
        if not active_control:
            continue
        frames = action.get("frames_advanced")
        if frames is None:
            frames = action.get("frames")
        try:
            advanced = max(0, int(frames or 0))
        except (TypeError, ValueError):
            advanced = 0
        active_steps += advanced
        if advanced:
            active_segments += 1

    terminal_prevented_counterexample = bool(
        objective.get("raw_satisfied") is True
        and objective.get("satisfied") is not True
        and objective.get("disqualified_by_terminal_event")
    )
    constraint_rules = (trial.get("constraints") or {}).get("checks") or []
    constraints_verified = not constraint_rules or (
        isinstance(constraints, dict)
        and isinstance(constraints.get("satisfied"), bool)
    )
    constraints_satisfied = constraints_verified and (constraints or {}).get("satisfied") is not False
    finished = bool(termination_reason and termination_reason != "not_stopped")
    valid = finished and constraints_satisfied and (
        terminal_prevented_counterexample
        or active_steps >= required_active_steps
    )
    if not constraints_verified:
        reason = "attempt_constraints_unverified"
    elif not constraints_satisfied:
        reason = "attempt_constraints_violated"
    elif terminal_prevented_counterexample:
        reason = "terminal_rule_prevented_counterexample"
    elif active_steps >= required_active_steps:
        reason = "active_search_completed"
    elif not finished:
        reason = "run_not_finished"
    else:
        reason = "insufficient_active_search"
    return {
        "valid": valid,
        "reason": reason,
        "active_steps": active_steps,
        "active_segments": active_segments,
        "required_active_steps": required_active_steps,
        "max_total_steps": max_total_steps,
        "termination_reason": termination_reason or "not_stopped",
        "terminal_prevented_counterexample": terminal_prevented_counterexample,
        "constraints_verified": constraints_verified,
        "constraints_satisfied": constraints_satisfied,
    }


def negative_search_repair_hint(evidence: dict[str, Any]) -> str:
    if evidence.get("reason") == "attempt_constraints_unverified":
        return "The result does not verify its attempt rules; rerun the test to collect complete evidence."
    if evidence.get("reason") == "attempt_constraints_violated":
        return "The counterexample search violated its attempt rules; rerun the test with those rules intact."
    if evidence.get("reason") == "run_not_finished":
        return "The counterexample search did not finish; rerun it to a bounded stopping point."
    return (
        "The counterexample search ended before enough active control evidence was collected; "
        "rerun the test."
    )


def _behavior_result_counts(
    *,
    results: list[dict[str, Any]],
    total: int,
    result_trial_ids: set[str],
) -> dict[str, int]:
    return {
        "passed": sum(item.get("status") == "passed" for item in results),
        "inconclusive": sum(item.get("status") == "inconclusive" for item in results),
        "failed": sum(item.get("status") == "failed" for item in results),
        "invalid_setup": sum(item.get("status") == "invalid_setup" for item in results),
        "errors": sum(
            item.get("status") not in CANONICAL_BEHAVIOR_RESULT_STATUSES
            or item.get("status") == "error"
            for item in results
        ),
        "not_run": max(0, total - len(result_trial_ids)),
    }


def _behavior_report_status(counts: dict[str, int]) -> str:
    if counts["failed"]:
        return "failed"
    if counts["inconclusive"] or counts["invalid_setup"]:
        return "needs_attention"
    if counts["errors"]:
        return "error"
    if counts["not_run"]:
        return "partial"
    return "passed"


def _behavior_report_next_action(counts: dict[str, int]) -> str:
    if counts["failed"]:
        return "Review the concrete counterexample and repair the environment."
    if counts["invalid_setup"]:
        return "Repair the invalid test setup before trusting the result."
    if counts["inconclusive"]:
        return "Rerun the tests that ended without enough evidence."
    if counts["errors"]:
        return "Retry the agent test after resolving the infrastructure error."
    if counts["not_run"]:
        return "Run the remaining agent tests for this scene snapshot."
    return "All agent tests passed for this scene snapshot."


def _coerce_bounded_runtime_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_runtime_float(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _single_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def write_behavior_trial_plan(scene_dir: Path, plan: dict[str, Any]) -> Path:
    path = scene_dir / ENV_BEHAVIOR_TRIAL_PLAN_FILENAME
    _atomic_write_json(path, plan)
    _register_metadata_artifact(scene_dir, "env_behavior_trial_plan", path, "behavior_plan")
    return path


def write_behavior_trial_report(scene_dir: Path, report: dict[str, Any]) -> Path:
    path = scene_dir / ENV_BEHAVIOR_TRIAL_REPORT_FILENAME
    _atomic_write_json(path, report)
    _register_metadata_artifact(scene_dir, "env_behavior_trial_report", path, "behavior_report")
    return path


def load_behavior_trial_plan(
    scene_dir: Path,
    *,
    current_spec: dict[str, Any] | EnvSpec3D | None = None,
) -> dict[str, Any] | None:
    plan = _read_optional_json(scene_dir / ENV_BEHAVIOR_TRIAL_PLAN_FILENAME)
    if not plan:
        return None
    spec_value: dict[str, Any] | EnvSpec3D | None = current_spec
    if spec_value is None:
        spec_value = _read_optional_json(scene_dir / "env_spec_3d.json")
        if spec_value is None:
            spec_value = _read_optional_json(scene_dir / "draft_env_spec_3d.json")
        if spec_value is None:
            return plan
    try:
        spec = spec_value if isinstance(spec_value, EnvSpec3D) else parse_env_spec_3d(spec_value)
    except Exception:
        return plan
    return upgrade_behavior_trial_plan(plan, spec=spec)


def upgrade_behavior_trial_plan(
    plan: dict[str, Any],
    *,
    spec: EnvSpec3D,
) -> dict[str, Any]:
    """Translate valid v1 behavior checks to the shared assertion schema.

    Invalid or ambiguous historical plans remain readable but carry migration
    issues so Studio can request regeneration rather than changing semantics.
    """

    trials = [trial for trial in plan.get("trials") or [] if isinstance(trial, dict)]
    if (
        trials
        and plan.get("schema_version") == ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION
        and plan.get("controller_version") == BEHAVIOR_CONTROLLER_VERSION
        and all(
            isinstance(check.get("predicate"), dict)
            for trial in trials
            for group in (trial.get("objective") or {}, trial.get("constraints") or {})
            for check in group.get("checks") or []
            if isinstance(check, dict)
        )
    ):
        try:
            for index, trial in enumerate(trials):
                _normalize_trial(index, trial, spec=spec)
        except (EnvBehaviorTrialError, ValueError) as exc:
            return {**plan, "migration_issues": [str(exc)]}
        return plan
    try:
        upgraded = normalize_behavior_trial_plan(
            env_id=str(plan.get("env_id") or spec.id),
            prompt=str(plan.get("prompt") or spec.description),
            operation_count=int(plan.get("operation_count") or 0),
            draft_spec=spec,
            fallback=bool(plan.get("fallback")),
            decision=str(plan.get("decision") or ("fallback" if plan.get("fallback") else "prompt_specific")),
            decision_reason=str(plan.get("decision_reason") or ""),
            source_turn_id=str(plan.get("source_turn_id") or ""),
            intent_summary=str(plan.get("intent_summary") or ""),
            trials=trials,
        )
    except (EnvBehaviorTrialError, ValueError) as exc:
        return {**plan, "migration_issues": [str(exc)]}
    for key in (
        "created_at",
        "dismissed",
        "dismissed_at",
        "dismissed_trial_ids",
        "preserved_from_turn_id",
        "preserved_from_prompt_hash",
    ):
        if key in plan:
            upgraded[key] = plan[key]
    upgraded["upgraded_from_schema_version"] = str(plan.get("schema_version") or "1.0")
    return upgraded


def load_behavior_trial_report(
    scene_dir: Path,
    *,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    report = _read_optional_json(scene_dir / ENV_BEHAVIOR_TRIAL_REPORT_FILENAME)
    if not report:
        return None
    return normalize_behavior_trial_report(
        report,
        plan=plan if plan is not None else load_behavior_trial_plan(scene_dir),
    )


def clear_behavior_trial_report(scene_dir: Path) -> None:
    path = scene_dir / ENV_BEHAVIOR_TRIAL_REPORT_FILENAME
    if path.is_file():
        path.unlink()


def behavior_trial_history(scene_dir: Path) -> list[dict[str, Any]]:
    root = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME
    if not root.is_dir():
        return []
    values: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _read_optional_json(run_dir / "manifest.json")
        report = _read_optional_json(run_dir / "report.json")
        normalized_report = (
            normalize_behavior_trial_report(report)
            if isinstance(report, dict)
            else None
        )
        if not manifest:
            continue
        source = normalized_report or manifest
        values.append(
            {
                "run_id": manifest.get("run_id") or run_dir.name,
                "created_at": manifest.get("created_at"),
                "completed_at": manifest.get("completed_at"),
                "status": source.get("status") or "error",
                "stale": bool(source.get("stale")),
                "trial_ids": manifest.get("trial_ids") or [],
                "summary": (normalized_report or {}).get("summary") or {},
            }
        )
    return sorted(values, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def active_behavior_runs(scene_dir: Path) -> list[dict[str, Any]]:
    root = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME
    if not root.is_dir():
        return []
    active: list[dict[str, Any]] = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = _read_optional_json(manifest_path)
        if (
            manifest
            and manifest.get("status") in {"preparing", "running", "replaying", "evaluating"}
            and _process_is_alive(manifest.get("pid"))
        ):
            active.append(manifest)
    return sorted(active, key=lambda item: str(item.get("created_at") or ""))


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


def _normalize_trial(index: int, raw: Any, *, spec: EnvSpec3D) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EnvBehaviorTrialError(f"trial {index + 1} must be an object")
    instruction = " ".join(str(raw.get("instruction") or raw.get("prompt") or "").split())
    if not instruction:
        raise EnvBehaviorTrialError(f"trial {index + 1} is missing instruction")
    expected = str(raw.get("expected_outcome") or raw.get("expected") or "should_succeed").strip().lower()
    if expected not in SUPPORTED_EXPECTED_OUTCOMES:
        raise EnvBehaviorTrialError(
            f"trial {index + 1} expected_outcome must be one of {sorted(SUPPORTED_EXPECTED_OUTCOMES)}"
        )
    severity = str(raw.get("severity") or "critical").strip().lower()
    if severity not in SUPPORTED_SEVERITIES:
        raise EnvBehaviorTrialError(f"trial {index + 1} severity must be critical or advisory")
    max_steps = _bounded_int(raw.get("max_steps", DEFAULT_MAX_STEPS), "max_steps", 60, MAX_STEPS)
    max_resets = _bounded_int(raw.get("max_resets", DEFAULT_MAX_RESETS), "max_resets", 0, MAX_RESETS)
    objective = _normalize_objective(
        raw.get("objective"),
        trial_index=index,
        field_name="objective",
        spec=spec,
    )
    constraints = _normalize_objective(
        raw.get("constraints"),
        trial_index=index,
        field_name="constraints",
        spec=spec,
        required=False,
    )
    duplicate_ids = {
        check["id"]
        for check in objective["checks"]
        if check["id"] in {constraint["id"] for constraint in constraints["checks"]}
    }
    if duplicate_ids:
        raise EnvBehaviorTrialError(
            "objective and constraint check ids must be unique: " + ", ".join(sorted(duplicate_ids))
        )
    if expected == "should_not_succeed":
        limit_only = [check["id"] for check in objective["checks"] if _is_limit_only_check(check)]
        if limit_only:
            raise EnvBehaviorTrialError(
                "should_not_succeed objectives must describe the prohibited counterexample positively; "
                "remove outcome limits and put only genuine attempt rules, such as no jumping, into "
                "constraints: " + ", ".join(limit_only)
            )
    allow_target_motion = raw.get("allow_target_motion", False)
    if not isinstance(allow_target_motion, bool):
        raise EnvBehaviorTrialError("allow_target_motion must be a boolean")
    return {
        "id": _normalize_id(raw.get("id") or f"trial_{index + 1}"),
        "instruction": instruction[:2400],
        "expected_outcome": expected,
        "severity": severity,
        "max_steps": max_steps,
        "max_resets": max_resets,
        "objective": objective,
        "constraints": constraints,
        "allow_target_motion": allow_target_motion,
    }


def _normalize_objective(
    raw: Any,
    *,
    trial_index: int,
    field_name: str,
    spec: EnvSpec3D,
    required: bool = True,
) -> dict[str, Any]:
    if raw is None and not required:
        return {"mode": "all", "checks": [], "ordered_check_ids": []}
    if not isinstance(raw, dict):
        raise EnvBehaviorTrialError(f"trial {trial_index + 1} {field_name} must be an object")
    checks = raw.get("conditions")
    if checks is None:
        checks = raw.get("checks")
    if not isinstance(checks, list) or (required and not checks):
        requirement = "a non-empty list" if required else "a list"
        raise EnvBehaviorTrialError(f"{field_name} conditions must be {requirement}")
    if not checks:
        return {"mode": "all", "checks": [], "ordered_check_ids": []}
    conditions: list[dict[str, Any]] = []
    legacy_by_id: dict[str, dict[str, Any]] = {}
    for check_index, check in enumerate(checks):
        if isinstance(check, dict) and isinstance(check.get("predicate"), dict):
            conditions.append(check)
            continue
        condition, legacy = _legacy_check_to_condition(
            check_index,
            check,
            spec=spec,
            field_name=field_name,
        )
        conditions.append(condition)
        legacy_by_id[condition["id"]] = legacy
    try:
        normalized = normalize_assertion_group(
            raw={
                "mode": raw.get("mode") or "all",
                "conditions": conditions,
                "ordered_condition_ids": (
                    raw.get("ordered_condition_ids")
                    or raw.get("ordered_check_ids")
                    or raw.get("ordered")
                    or []
                ),
            },
            spec=env_spec_to_dict(spec),
            required=required,
            field_name=field_name,
        )
    except TaskDefinitionError as exc:
        raise EnvBehaviorTrialError(str(exc)) from exc
    for condition in normalized["checks"]:
        predicate = condition.get("predicate") or {}
        condition["predicate_type"] = str(predicate.get("type") or "")
        if legacy := legacy_by_id.get(str(condition.get("id"))):
            condition["type"] = str(legacy.get("type") or predicate.get("type") or "")
            condition["legacy_type"] = legacy.get("type")
            for key, value in legacy.items():
                if key not in {"id", "description", "type"}:
                    condition.setdefault(key, value)
        else:
            condition["type"] = str(predicate.get("type") or "")
    return normalized


def _legacy_check_to_condition(
    index: int,
    raw: Any,
    *,
    spec: EnvSpec3D,
    field_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy = _normalize_behavior_check(index, raw)
    check_type = legacy["type"]
    agent = {"semantic_type": "agent"}
    predicate: dict[str, Any]
    temporal = "eventually"
    condition_bounds: dict[str, Any] = {}

    if check_type == "agent_displacement":
        predicate = {
            "type": "displacement",
            "subject": agent,
            "metric": "maximum",
            "space": legacy.get("space") or "xy",
            **_legacy_number_bounds(legacy, "min_distance", "max_distance"),
        }
        temporal = _legacy_bounded_temporal(legacy, "min_distance", "max_distance")
    elif check_type == "agent_height_gain":
        predicate = {
            "type": "axis_delta",
            "subject": agent,
            "axis": "z",
            "metric": "maximum",
            **_legacy_number_bounds(legacy, "min_gain", "max_gain"),
        }
        temporal = _legacy_bounded_temporal(legacy, "min_gain", "max_gain")
    elif check_type == "jump_count":
        predicate = {
            "type": "jump_count",
            **_legacy_number_bounds(legacy, "min_count", "max_count"),
        }
        temporal = _legacy_bounded_temporal(legacy, "min_count", "max_count")
    elif check_type == "zone_entry":
        targets = select_scene_objects(scene_objects(spec), legacy["selector"])
        if any(target.body_type != "sensor" for target in targets):
            raise EnvBehaviorTrialError(
                f"{field_name} legacy zone_entry selector must match sensor zones. "
                "To move an object into a region, use overlap with an explicit "
                "subject and target."
            )
        predicate = {
            "type": "overlap",
            "subject": agent,
            "target": legacy["selector"],
        }
        temporal = "count"
        condition_bounds = _legacy_count_bounds(legacy)
    elif check_type == "contact_count":
        predicate = {
            "type": "contact",
            "subject": agent,
            "target": legacy["selector"],
        }
        temporal = "count"
        condition_bounds = _legacy_count_bounds(legacy)
    elif check_type == "object_displacement":
        predicate = {
            "type": "displacement",
            "subject": legacy["selector"],
            "metric": legacy.get("metric") or "maximum",
            "space": "xyz",
            **_legacy_number_bounds(legacy, "min_distance", "max_distance"),
        }
        temporal = _legacy_bounded_temporal(legacy, "min_distance", "max_distance")
    elif check_type == "agent_relation":
        predicate = {
            "type": "relation",
            "subject": agent,
            "target": legacy["target"],
            "relation": legacy["relation"],
            "margin": legacy.get("margin", 0.0),
        }
        if "min_distance" in legacy:
            predicate["min_distance"] = legacy["min_distance"]
        if "max_distance" in legacy:
            predicate["max_distance"] = legacy["max_distance"]
        temporal = "at_end" if legacy.get("when") == "final" else "eventually"
    elif check_type == "mechanism_state":
        predicate = {
            "type": "mechanism_state",
            "mechanism_id": legacy["mechanism_id"],
            "state": legacy["state"],
        }
        if "min_progress" in legacy:
            predicate["min_progress"] = legacy["min_progress"]
    elif check_type == "terminal_event":
        predicate = {"type": "terminal_event", "event": legacy["event"]}
        temporal = "count"
        condition_bounds = _legacy_count_bounds(legacy)
    elif check_type == "attempt_reset":
        predicate = {"type": "reset_event", "reason": legacy["reason"]}
        temporal = "count"
        condition_bounds = _legacy_count_bounds(legacy)
    else:  # pragma: no cover - legacy normalization rejects unknown types
        raise EnvBehaviorTrialError(f"cannot translate legacy behavior check {check_type!r}")

    return (
        {
            "id": legacy["id"],
            "description": legacy.get("description") or check_type.replace("_", " "),
            "temporal": temporal,
            "predicate": predicate,
            **condition_bounds,
        },
        legacy,
    )


def _legacy_number_bounds(
    value: dict[str, Any],
    minimum_key: str,
    maximum_key: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if minimum_key in value:
        result["min_value"] = value[minimum_key]
    if maximum_key in value:
        result["max_value"] = value[maximum_key]
    return result


def _legacy_count_bounds(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "min_count" in value:
        result["min_count"] = value["min_count"]
    if "max_count" in value:
        result["max_count"] = value["max_count"]
    return result


def _legacy_bounded_temporal(
    value: dict[str, Any],
    minimum_key: str,
    maximum_key: str,
) -> str:
    if maximum_key in value and minimum_key not in value:
        return "always"
    return "eventually"


def _is_limit_only_check(check: dict[str, Any]) -> bool:
    temporal = str(check.get("temporal") or "eventually")
    predicate = check.get("predicate") or {}
    if temporal in {"always", "never"}:
        return True
    if temporal == "count":
        return "max_count" in check and int(check.get("min_count") or 0) <= 0
    return "max_value" in predicate and "min_value" not in predicate


def _normalize_behavior_check(index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EnvBehaviorTrialError(f"objective check {index + 1} must be an object")
    check_type = str(raw.get("type") or "").strip().lower()
    if check_type not in SUPPORTED_BEHAVIOR_CHECK_TYPES:
        raise EnvBehaviorTrialError(
            f"unsupported behavior check {check_type!r}; expected one of {sorted(SUPPORTED_BEHAVIOR_CHECK_TYPES)}"
        )
    check: dict[str, Any] = {
        "id": _normalize_id(raw.get("id") or f"{check_type}_{index + 1}"),
        "type": check_type,
        "description": " ".join(str(raw.get("description") or "").split())[:500],
    }
    if check_type == "agent_displacement":
        check.update(_distance_bounds(raw, default_min=1.0))
        space = str(raw.get("space") or "xy").lower()
        if space not in {"xy", "xyz"}:
            raise EnvBehaviorTrialError("agent_displacement space must be xy or xyz")
        check["space"] = space
    elif check_type == "agent_height_gain":
        check.update(_number_bounds(raw, min_key="min_gain", max_key="max_gain", default_min=0.5))
    elif check_type == "jump_count":
        check.update(_count_bounds(raw, default_min=1))
    elif check_type in {"zone_entry", "contact_count"}:
        check["selector"] = normalize_selector(raw.get("selector") or raw.get("target"), "selector")
        check.update(_count_bounds(raw, default_min=1))
    elif check_type == "object_displacement":
        check["selector"] = normalize_selector(raw.get("selector") or raw.get("target"), "selector")
        check.update(_distance_bounds(raw, default_min=0.25))
        check["metric"] = str(raw.get("metric") or "maximum").lower()
        if check["metric"] not in {"maximum", "final"}:
            raise EnvBehaviorTrialError("object_displacement metric must be maximum or final")
    elif check_type == "agent_relation":
        relation = str(raw.get("relation") or "").strip().lower()
        if relation not in SUPPORTED_RELATIONS:
            raise EnvBehaviorTrialError(
                f"agent_relation relation must be one of {sorted(SUPPORTED_RELATIONS)}"
            )
        check["target"] = normalize_selector(raw.get("target") or raw.get("selector"), "target")
        check["relation"] = relation
        check["when"] = str(raw.get("when") or "ever").lower()
        if check["when"] not in SUPPORTED_CHECK_TIMES:
            raise EnvBehaviorTrialError("agent_relation when must be ever or final")
        check["margin"] = _finite_float(raw.get("margin", 0.0), "margin")
        if relation == "near":
            check["max_distance"] = _positive_float(raw.get("max_distance", 2.0), "max_distance")
        elif relation == "far_from":
            check["min_distance"] = _positive_float(raw.get("min_distance", 2.0), "min_distance")
    elif check_type == "mechanism_state":
        check["mechanism_id"] = _normalize_id(raw.get("mechanism_id"))
        state = str(raw.get("state") or "active").strip().lower()
        if state not in {"active", "open"}:
            raise EnvBehaviorTrialError("mechanism_state state must be active or open")
        check["state"] = state
        if state == "open":
            check["min_progress"] = _finite_float(raw.get("min_progress", 0.9), "min_progress")
            if not 0 < check["min_progress"] <= 1:
                raise EnvBehaviorTrialError("mechanism_state min_progress must be in (0, 1]")
    elif check_type == "terminal_event":
        event = str(raw.get("event") or raw.get("outcome") or "").strip().lower()
        if event not in SUPPORTED_TERMINAL_EVENTS:
            raise EnvBehaviorTrialError(
                "terminal_event event must be one of " + str(sorted(SUPPORTED_TERMINAL_EVENTS))
            )
        check["event"] = event
        check.update(_count_bounds(raw, default_min=1))
    elif check_type == "attempt_reset":
        reason = str(raw.get("reason") or "any").strip().lower()
        if reason not in SUPPORTED_RESET_REASONS:
            raise EnvBehaviorTrialError(
                "attempt_reset reason must be one of " + str(sorted(SUPPORTED_RESET_REASONS))
            )
        check["reason"] = reason
        check.update(_count_bounds(raw, default_min=1))
    return check


def _distance_bounds(raw: dict[str, Any], *, default_min: float) -> dict[str, float]:
    return _number_bounds(
        raw,
        min_key="min_distance",
        max_key="max_distance",
        default_min=default_min,
    )


def _number_bounds(
    raw: dict[str, Any],
    *,
    min_key: str,
    max_key: str,
    default_min: float,
) -> dict[str, float]:
    minimum = raw.get(min_key, raw.get("min"))
    maximum = raw.get(max_key, raw.get("max"))
    if minimum is None and maximum is None:
        minimum = default_min
    result: dict[str, float] = {}
    if minimum is not None:
        result[min_key] = _non_negative_float(minimum, min_key)
    if maximum is not None:
        result[max_key] = _non_negative_float(maximum, max_key)
    if min_key in result and max_key in result and result[min_key] > result[max_key]:
        raise EnvBehaviorTrialError(f"{min_key} cannot exceed {max_key}")
    return result


def _count_bounds(raw: dict[str, Any], *, default_min: int) -> dict[str, int]:
    minimum = raw.get("min_count", raw.get("min"))
    maximum = raw.get("max_count", raw.get("max"))
    if minimum is None and maximum is None:
        minimum = default_min
    result: dict[str, int] = {}
    if minimum is not None:
        result["min_count"] = _bounded_int(minimum, "min_count", 0, 100000)
    if maximum is not None:
        result["max_count"] = _bounded_int(maximum, "max_count", 0, 100000)
    if "min_count" in result and "max_count" in result and result["min_count"] > result["max_count"]:
        raise EnvBehaviorTrialError("min_count cannot exceed max_count")
    return result


def _agent_objects(spec: EnvSpec3D) -> list[Any]:
    return [obj for obj in spec.objects if obj.semantic_type.lower() == "agent"]


def _optional_spec(value: dict[str, Any] | EnvSpec3D | None) -> EnvSpec3D | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, EnvSpec3D) else parse_env_spec_3d(value)
    except Exception:
        return None


def _normalize_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    if not cleaned:
        raise EnvBehaviorTrialError("id cannot be blank")
    return cleaned[:80]


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise EnvBehaviorTrialError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EnvBehaviorTrialError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise EnvBehaviorTrialError(f"{name} must be between {minimum} and {maximum}")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EnvBehaviorTrialError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise EnvBehaviorTrialError(f"{name} must be finite")
    return result


def _non_negative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise EnvBehaviorTrialError(f"{name} cannot be negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise EnvBehaviorTrialError(f"{name} must be positive")
    return result


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


def _register_metadata_artifact(scene_dir: Path, key: str, path: Path, role: str) -> None:
    metadata_path = scene_dir / "metadata.json"
    metadata = _read_optional_json(metadata_path)
    if not metadata or not path.is_file():
        return
    artifacts = metadata.get("artifacts") if isinstance(metadata.get("artifacts"), dict) else {}
    data = path.read_bytes()
    artifacts[key] = {
        "path": path.name,
        "role": role,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    metadata["artifacts"] = artifacts
    _atomic_write_json(metadata_path, metadata)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
