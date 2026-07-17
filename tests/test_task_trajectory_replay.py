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
def test_task_activity_timeline_aligns_notes_and_actions_to_simulation_steps() -> None:
    root = Path(__file__).parents[1]
    module_url = (
        root / "environment_generation" / "studio_web" / "task_trajectory_replay.js"
    ).as_uri()
    script = f"""
      import {{ normalizeTaskActivityTimeline, taskActivityAtStep }} from {json.dumps(module_url)};
      const timeline = normalizeTaskActivityTimeline([
        {{ type: "phase", message: "running" }},
        {{ type: "agent_message", message: "I will inspect the route." }},
        {{ type: "tool_result", message: "Observed.", summary: {{ step: 24 }} }},
        {{ type: "agent_message", message: "The goal is ahead." }},
        {{ type: "tool_result", message: "Advanced.", step: 60 }},
      ]);
      console.log(JSON.stringify({{
        steps: timeline.map((event) => event.step),
        at23: taskActivityAtStep(timeline, 23).map((event) => event.message),
        at24: taskActivityAtStep(timeline, 24).map((event) => event.message),
        at60: taskActivityAtStep(timeline, 60).map((event) => event.message),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "steps": [0, 0, 24, 24, 60],
        "at23": ["running", "I will inspect the route."],
        "at24": [
            "running",
            "I will inspect the route.",
            "Observed.",
            "The goal is ahead.",
        ],
        "at60": [
            "running",
            "I will inspect the route.",
            "Observed.",
            "The goal is ahead.",
            "Advanced.",
        ],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_task_trajectory_runs_merge_latest_activity_and_sort_newest_first() -> None:
    root = Path(__file__).parents[1]
    module_url = (
        root / "environment_generation" / "studio_web" / "task_trajectory_replay.js"
    ).as_uri()
    script = f"""
      import {{ taskTrajectoryRuns }} from {json.dumps(module_url)};
      const runs = taskTrajectoryRuns({{
        run_summaries: [
          {{ run_id: "run-0001-old", completed_at: "2026-01-01T00:00:00Z", trajectory_url: "/one.json", passed: false }},
          {{ run_id: "run-0002-new", completed_at: "2026-01-02T00:00:00Z", trajectory_url: "/two.json", passed: true }},
          {{ run_id: "run-0003-error", completed_at: "2026-01-03T00:00:00Z", status: "error" }},
        ],
        latest_run: {{
          run_id: "run-0002-new",
          trajectory_url: "/two.json",
          activity: [{{ type: "agent_message", message: "Done." }}],
          passed: true,
        }},
      }});
      console.log(JSON.stringify(runs.map((run) => ({{
        id: run.run_id,
        number: run.run_number,
        activity: run.activity?.length || 0,
      }}))));
    """

    result = _run_node(root, script)

    assert result == [
        {"id": "run-0002-new", "number": 2, "activity": 1},
        {"id": "run-0001-old", "number": 1, "activity": 0},
    ]
