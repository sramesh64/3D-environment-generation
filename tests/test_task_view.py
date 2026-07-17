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
def test_task_rows_show_each_condition_once_without_group_or_temporal_metadata() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "task_view.js").as_uri()
    script = f"""
      import {{ buildTaskTestRows }} from {json.dumps(module_url)};
      const task = {{ tests: [{{
        id: "reach_goal",
        description: "The robot eventually overlaps the charging goal sensor.",
        source: "codex",
        conditions: [{{
          id: "enter_goal",
          description: "The robot enters the charging goal region.",
          temporal: "eventually",
          predicate: {{ type: "overlap" }},
        }}],
      }}] }};
      const report = {{ tests: [{{
        id: "reach_goal",
        conditions: [{{ id: "enter_goal", passed: true }}],
      }}] }};
      console.log(JSON.stringify({{ rows: buildTaskTestRows(task, report), source: task.tests[0].source }}));
    """

    result = _run_node(root, script)

    assert result["rows"] == [
        {
            "id": "enter_goal",
            "testId": "reach_goal",
            "description": "The robot enters the charging goal region.",
            "passed": True,
        }
    ]
    assert "eventually" not in result["rows"][0]["description"].lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_task_rows_fall_back_to_group_description_when_condition_has_no_label() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "task_view.js").as_uri()
    script = f"""
      import {{ buildTaskTestRows }} from {json.dumps(module_url)};
      console.log(JSON.stringify(buildTaskTestRows({{
        tests: [{{
          id: "stay_safe",
          description: "Stay inside the courtyard.",
          conditions: [{{ id: "inside", predicate: {{ type: "in_bounds" }} }}],
        }}],
      }})));
    """

    result = _run_node(root, script)

    assert result == [
        {
            "id": "inside",
            "testId": "stay_safe",
            "description": "Stay inside the courtyard.",
            "passed": None,
        }
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_task_result_view_is_neutral_until_a_run_or_oracle_is_selected() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "task_view.js").as_uri()
    script = f"""
      import {{ buildTaskResultView }} from {json.dumps(module_url)};
      const task = {{ tests: [{{
        id: "route",
        conditions: [
          {{ id: "switch", description: "Press the switch." }},
          {{ id: "goal", description: "Reach the goal." }},
        ],
      }}] }};
      const mixed = {{ tests: [{{
        id: "route",
        conditions: [
          {{ id: "switch", passed: true }},
          {{ id: "goal", passed: false }},
        ],
      }}] }};
      const passed = {{ tests: [{{
        id: "route",
        conditions: [
          {{ id: "switch", passed: true }},
          {{ id: "goal", passed: true }},
        ],
      }}] }};
      console.log(JSON.stringify({{
        neutral: buildTaskResultView(task),
        run: buildTaskResultView(task, {{ kind: "run", label: "Run 3", report: mixed }}),
        oracle: buildTaskResultView(task, {{ kind: "oracle", label: "Oracle", report: passed }}),
      }}));
    """

    result = _run_node(root, script)

    assert result["neutral"]["summary"] == "2 tests"
    assert [row["passed"] for row in result["neutral"]["rows"]] == [None, None]
    assert result["run"]["summary"] == "Run 3 · 1/2 passed"
    assert [row["passed"] for row in result["run"]["rows"]] == [True, False]
    assert result["oracle"]["summary"] == "Oracle · 2/2 passed"
    assert [row["passed"] for row in result["oracle"]["rows"]] == [True, True]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_recording_pacing_tracks_elapsed_time_and_caps_stalls() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "task_recording.js").as_uri()
    script = f"""
      import {{ recordingFrameCount, recordingReportRequested }} from {json.dumps(module_url)};
      console.log(JSON.stringify({{
        normal: recordingFrameCount({{ elapsedMs: 40, timestep: 0.01 }}),
        delayed: recordingFrameCount({{ elapsedMs: 75, timestep: 0.01 }}),
        capped: recordingFrameCount({{ elapsedMs: 1000, timestep: 0.01 }}),
        minimum: recordingFrameCount({{ elapsedMs: 0, timestep: 0.01 }}),
        reports: {{
          whileMoving: recordingReportRequested({{ activeInput: true, idleTicks: 30, idleLimit: 30 }}),
          whileSettling: recordingReportRequested({{ activeInput: false, idleTicks: 12, idleLimit: 30 }}),
          whenPaused: recordingReportRequested({{ activeInput: false, idleTicks: 30, idleLimit: 30 }}),
          afterPause: recordingReportRequested({{ activeInput: false, idleTicks: 31, idleLimit: 30 }}),
        }},
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "normal": 4,
        "delayed": 8,
        "capped": 10,
        "minimum": 1,
        "reports": {
            "whileMoving": False,
            "whileSettling": False,
            "whenPaused": True,
            "afterPause": False,
        },
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_finish_recording_stays_clickable_during_a_step_request() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "task_recording.js").as_uri()
    script = f"""
      import {{ recordingFinishIntent, recordingFinishView }} from {json.dumps(module_url)};
      console.log(JSON.stringify({{
        empty: recordingFinishView({{ active: true, hasActions: false, requestInFlight: true }}),
        stepping: recordingFinishView({{ active: true, hasActions: true, requestInFlight: true }}),
        queued: recordingFinishView({{ active: true, hasActions: true, finishPending: true }}),
        finishing: recordingFinishView({{ active: true, hasActions: true, finishing: true }}),
        queuedIntent: recordingFinishIntent({{
          active: true,
          sessionId: "session",
          hasActions: true,
          requestInFlight: true,
        }}),
        immediateIntent: recordingFinishIntent({{
          active: true,
          sessionId: "session",
          hasActions: true,
          requestInFlight: false,
        }}),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "empty": {"disabled": True, "label": "Finish Recording"},
        "stepping": {"disabled": False, "label": "Finish Recording"},
        "queued": {"disabled": True, "label": "Finishing..."},
        "finishing": {"disabled": True, "label": "Finishing..."},
        "queuedIntent": {"accepted": True, "startImmediately": False},
        "immediateIntent": {"accepted": True, "startImmediately": True},
    }
