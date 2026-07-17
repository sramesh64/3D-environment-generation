"""Shared episode decisions for interactive and evaluated simulator runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


EpisodeOutcome = Literal["continue", "succeeded", "failed"]


@dataclass(frozen=True)
class EpisodeDecision:
    outcome: EpisodeOutcome
    reason: str = ""

    @property
    def terminal(self) -> bool:
        return self.outcome != "continue"


def decide_episode(
    *,
    safety_failure: str = "",
    explicit_failure: bool = False,
    objective_satisfied: bool = False,
    failure_reason: str = "objective_failed",
    success_reason: str = "objective_satisfied",
) -> EpisodeDecision:
    """Apply the same terminal precedence in every simulator runner.

    Semantic zone contact is intentionally absent from this API. Callers pass a
    failure or success only when their explicit evaluator assigns that meaning.
    """

    if safety_failure:
        return EpisodeDecision("failed", safety_failure)
    if explicit_failure:
        return EpisodeDecision("failed", failure_reason)
    if objective_satisfied:
        return EpisodeDecision("succeeded", success_reason)
    return EpisodeDecision("continue")


def assertion_report_irreversibly_failed(report: dict[str, Any]) -> bool:
    """Return whether continued frames cannot make the assertion report pass."""

    return any(
        _assertion_group_irreversibly_failed(test)
        for test in report.get("tests") or []
        if isinstance(test, dict)
    )


def _assertion_group_irreversibly_failed(group: dict[str, Any]) -> bool:
    conditions = [
        condition
        for condition in (group.get("conditions") or group.get("checks") or [])
        if isinstance(condition, dict)
    ]
    if not conditions:
        return False
    failures = [_condition_irreversibly_failed(condition) for condition in conditions]
    return any(failures) if str(group.get("mode") or "all") == "all" else all(failures)


def _condition_irreversibly_failed(condition: dict[str, Any]) -> bool:
    if bool(condition.get("passed")):
        return False
    temporal = str(condition.get("temporal") or "eventually")
    if temporal in {"always", "never"}:
        return True
    if temporal != "count" or condition.get("max_count") is None:
        return False
    metrics = condition.get("metrics") or {}
    return int(metrics.get("transition_count") or 0) > int(condition["max_count"])
