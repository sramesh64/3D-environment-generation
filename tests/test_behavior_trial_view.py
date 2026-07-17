from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _run_node(root: Path, script: str) -> dict[str, object]:
    completed = subprocess.run(
        [shutil.which("node") or "node", "--input-type=module", "-e", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_active_server_run_keeps_other_results_visible_and_uses_manifest_trials() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorRunView }} from {json.dumps(module_url)};
      const previous = {{ run_id: "old", results: [{{ trial_id: "climb", status: "inconclusive" }}] }};
      const result = buildBehaviorRunView({{
        summary: {{ active_run: {{ run_id: "new", status: "running", trial_ids: ["climb"] }} }},
        report: previous,
        trialIds: ["climb", "push"],
      }});
      console.log(JSON.stringify(result));
    """

    result = _run_node(root, script)

    assert result["active"] is True
    assert result["activeRunId"] == "new"
    assert result["activeTrialIds"] == ["climb"]
    assert result["runningCount"] == 1
    assert result["displayReport"] == {
        "run_id": "old",
        "results": [{"trial_id": "climb", "status": "inconclusive"}],
    }
    assert result["retainedReport"] == {
        "run_id": "old",
        "results": [{"trial_id": "climb", "status": "inconclusive"}],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_local_run_has_pending_state_before_server_manifest_arrives() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorRunView }} from {json.dumps(module_url)};
      const result = buildBehaviorRunView({{
        summary: {{ status: "needs_attention" }},
        report: {{ run_id: "old" }},
        trialIds: ["climb", "push"],
        localRunning: true,
        requestedTrialIds: ["climb"],
      }});
      console.log(JSON.stringify(result));
    """

    result = _run_node(root, script)

    assert result["active"] is True
    assert result["activeTrialIds"] == ["climb"]
    assert result["displayReport"] == {"run_id": "old"}
    assert result["retainedReport"] == {"run_id": "old"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_multiple_server_and_local_runs_union_their_active_test_ids() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorRunView }} from {json.dumps(module_url)};
      const result = buildBehaviorRunView({{
        summary: {{
          active_runs: [
            {{ run_id: "run-a", status: "running", trial_ids: ["climb"] }},
            {{ run_id: "run-b", status: "running", trial_ids: ["push"] }},
          ],
        }},
        trialIds: ["climb", "push", "jump"],
        localRunning: true,
        requestedTrialIds: ["jump"],
      }});
      console.log(JSON.stringify(result));
    """

    result = _run_node(root, script)

    assert result["active"] is True
    assert result["activeTrialIds"] == ["climb", "push", "jump"]
    assert result["runningCount"] == 3
    assert [run["run_id"] for run in result["activeRuns"]] == ["run-a", "run-b"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_inactive_view_retains_current_report() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorRunView }} from {json.dumps(module_url)};
      const report = {{ run_id: "current", status: "passed" }};
      console.log(JSON.stringify(buildBehaviorRunView({{
        summary: {{ status: "passed" }},
        report,
        trialIds: ["climb"],
      }})));
    """

    result = _run_node(root, script)

    assert result["active"] is False
    assert result["activeTrialIds"] == []
    assert result["displayReport"] == {"run_id": "current", "status": "passed"}
    assert result["retainedReport"] == {"run_id": "current", "status": "passed"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_header_collapses_successful_trial_counts() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorHeaderView }} from {json.dumps(module_url)};
      console.log(JSON.stringify(buildBehaviorHeaderView({{
        summary: {{ status: "passed", label: "Behavior trials: passed" }},
        report: {{ summary: {{ passed: 2, inconclusive: 0, failed: 0 }} }},
        trialCount: 2,
      }})));
    """

    result = _run_node(root, script)

    assert result == {
        "label": "All 2 tests passed",
        "stats": [{"label": "Passed", "value": "2/2", "tone": "good"}],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_header_omits_zero_status_counts_for_mixed_results() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorHeaderView }} from {json.dumps(module_url)};
      console.log(JSON.stringify(buildBehaviorHeaderView({{
        summary: {{ status: "needs_attention", label: "Behavior trials: needs attention" }},
        report: {{ summary: {{ passed: 1, inconclusive: 1, failed: 0 }} }},
        trialCount: 2,
      }})));
    """

    result = _run_node(root, script)

    assert result["label"] == "needs attention"
    assert result["stats"] == [
        {"label": "Passed", "value": "1", "tone": "good"},
        {"label": "Inconclusive", "value": "1", "tone": "warn"},
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_running_header_uses_plain_language_without_duplicate_stats() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorHeaderView }} from {json.dumps(module_url)};
      console.log(JSON.stringify(buildBehaviorHeaderView({{
        trialCount: 3,
        active: true,
        runningCount: 1,
      }})));
    """

    result = _run_node(root, script)

    assert result == {"label": "1 test running", "stats": []}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_negative_test_outcomes_use_plain_language_without_false_failed_checks() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorOutcomeView }} from {json.dumps(module_url)};
      console.log(JSON.stringify({{
        passed: buildBehaviorOutcomeView({{
          status: "passed",
          expectedOutcome: "should_not_succeed",
        }}),
        failed: buildBehaviorOutcomeView({{
          status: "failed",
          expectedOutcome: "should_not_succeed",
        }}),
        positive: buildBehaviorOutcomeView({{
          status: "passed",
          expectedOutcome: "should_succeed",
        }}),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "passed": {
            "tone": "passed",
            "label": "The prohibited behavior was not demonstrated during a valid bounded search.",
        },
        "failed": {
            "tone": "failed",
            "label": "The agent demonstrated the prohibited behavior.",
        },
        "positive": None,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_check_labels_and_metrics_expand_generic_predicate_output() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ describeBehaviorCheck, formatBehaviorMetric }} from {json.dumps(module_url)};
      const contact = {{
        id: "crate_contacts_ramp",
        description: "contact",
        predicate: {{
          type: "contact",
          subject: {{ id: "second_stackable_crate" }},
          target: {{ id: "box_stacking_ramp" }},
        }},
      }};
      const relation = {{
        id: "agent_clears_wall",
        description: "relation",
        predicate: {{
          type: "relation",
          subject: {{ id: "blue_robot" }},
          target: {{ id: "middle_wall_left_segment" }},
          relation: "above",
        }},
      }};
      console.log(JSON.stringify({{
        contactLabel: describeBehaviorCheck(contact),
        contactMetric: formatBehaviorMetric({{ transition_count: 42 }}, contact),
        relationLabel: describeBehaviorCheck(relation),
        relationMetric: formatBehaviorMetric({{ transition_count: 0 }}, relation),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "contactLabel": "Second stackable crate makes contact with box stacking ramp.",
        "contactMetric": "42 contact events",
        "relationLabel": "Blue robot is above middle wall left segment.",
        "relationMetric": "0 matches",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_historical_milestones_use_the_check_completed_at_that_step() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_trial_view.js").as_uri()
    script = f"""
      import {{ describeBehaviorMilestone }} from {json.dumps(module_url)};
      const objective = {{ checks: [
        {{
          id: "touch_ramp",
          description: "contact",
          first_satisfied_step: 272,
          predicate: {{
            type: "contact",
            subject: {{ id: "second_stackable_crate" }},
            target: {{ id: "box_stacking_ramp" }},
          }},
        }},
        {{
          id: "stack_crates",
          description: "relation",
          first_satisfied_step: 1773,
          predicate: {{
            type: "relation",
            subject: {{ id: "second_stackable_crate" }},
            target: {{ id: "pushable_supply_crate" }},
            relation: "on_surface",
          }},
        }},
      ] }};
      console.log(JSON.stringify({{
        contact: describeBehaviorMilestone({{ label: "Completed: contact", step: 273 }}, objective),
        relation: describeBehaviorMilestone({{ label: "Completed: relation", step: 1773 }}, objective),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "contact": "Completed: Second stackable crate makes contact with box stacking ramp",
        "relation": "Completed: Second stackable crate is on top of pushable supply crate",
    }
