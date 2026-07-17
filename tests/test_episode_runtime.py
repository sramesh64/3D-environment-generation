from __future__ import annotations

from environment_generation.episode_runtime import (
    assertion_report_irreversibly_failed,
    decide_episode,
)


def test_episode_decision_uses_explicit_policy_not_semantic_zone_names() -> None:
    assert decide_episode().outcome == "continue"
    assert decide_episode(objective_satisfied=True).outcome == "succeeded"
    assert decide_episode(explicit_failure=True).outcome == "failed"
    safety = decide_episode(
        safety_failure="out_of_bounds",
        explicit_failure=False,
        objective_satisfied=True,
    )
    assert (safety.outcome, safety.reason) == ("failed", "out_of_bounds")


def test_only_temporally_irreversible_assertion_failures_stop_early() -> None:
    eventually_unmet = {
        "tests": [{"mode": "all", "conditions": [{"passed": False, "temporal": "eventually"}]}]
    }
    never_violated = {
        "tests": [{"mode": "all", "conditions": [{"passed": False, "temporal": "never"}]}]
    }
    any_group_with_one_remaining_path = {
        "tests": [{
            "mode": "any",
            "conditions": [
                {"passed": False, "temporal": "never"},
                {"passed": False, "temporal": "eventually"},
            ],
        }]
    }

    assert assertion_report_irreversibly_failed(eventually_unmet) is False
    assert assertion_report_irreversibly_failed(never_violated) is True
    assert assertion_report_irreversibly_failed(any_group_with_one_remaining_path) is False
