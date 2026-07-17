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
def test_milestone_view_frames_agent_and_current_objective() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorMilestoneView }} from {json.dumps(module_url)};
      const visualScene = {{
        camera: {{ azimuth: -42, min_distance: 8.4, max_distance: 39.6 }},
        objects: [
          {{ source_id: "robot", semantic_type: "agent", position: [6, 4, 0.55], size: [0.7, 0.7, 1.1] }},
          {{ source_id: "goal", semantic_type: "goal", position: [-6, -3, 0.6], size: [1.4, 1.4, 1.2] }},
        ],
      }};
      const evidenceFrame = {{
        agent: {{ id: "robot", position: [2, 1, 0.55] }},
        objective_focus: {{ target_ids: ["goal"] }},
        navigation: {{ primary_target_id: "goal" }},
      }};
      const trajectoryFrame = {{ objects: [{{ id: "robot", position: [2, 1, 0.55] }}] }};
      console.log(JSON.stringify(buildBehaviorMilestoneView({{ visualScene, evidenceFrame, trajectoryFrame }})));
    """

    result = _run_node(root, script)

    assert result["agent_id"] == "robot"
    assert result["focus_ids"] == ["goal"]
    assert result["target"] == pytest.approx([-2.3, -1.3, 0.65], abs=1e-3)
    assert result["distance"] == pytest.approx(16.456, abs=1e-3)
    assert result["azimuth"] == -42
    assert result["elevation"] == 54


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_milestone_view_uses_event_target_after_objective_completion() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorMilestoneView }} from {json.dumps(module_url)};
      const visualScene = {{
        camera: {{ azimuth: 25 }},
        objects: [
          {{ source_id: "robot", semantic_type: "agent", position: [0, 0, 0.55], size: [0.7, 0.7, 1.1] }},
          {{ source_id: "target_region", semantic_type: "target_region", position: [4, 2, 0.1], size: [2, 2, 0.2] }},
        ],
      }};
      const evidenceFrame = {{
        agent: {{ id: "robot", position: [3.8, 1.9, 0.55] }},
        objective_focus: {{ satisfied: true, target_ids: [] }},
        navigation: {{ available: false, status: "objective_satisfied" }},
        events: [{{ type: "zone_entered", object_id: "target_region" }}],
      }};
      console.log(JSON.stringify(buildBehaviorMilestoneView({{ visualScene, evidenceFrame, trajectoryFrame: {{}} }})));
    """

    result = _run_node(root, script)

    assert result["focus_ids"] == ["target_region"]
    assert result["agent_id"] == "robot"
    assert result["distance"] < 10
    assert result["azimuth"] == 25


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_milestone_view_supports_multiple_targets_and_agent_only_trials() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorMilestoneView }} from {json.dumps(module_url)};
      const visualScene = {{
        camera: {{ min_distance: 8.4, max_distance: 30 }},
        objects: [
          {{ source_id: "robot", semantic_type: "agent", position: [0, 0, 0.5], size: [0.7, 0.7, 1] }},
          {{ source_id: "crate_a", position: [-3, 2, 0.5], size: [1, 1, 1] }},
          {{ source_id: "crate_b", position: [4, -2, 0.5], size: [1, 1, 1] }},
        ],
      }};
      const multi = buildBehaviorMilestoneView({{
        visualScene,
        evidenceFrame: {{
          agent: {{ id: "robot" }},
          objective_focus: {{ subject_ids: ["crate_a"], target_ids: ["crate_b"] }},
        }},
        trajectoryFrame: {{ objects: [{{ id: "robot", position: [1, 1, 0.5] }}] }},
      }});
      const agentOnly = buildBehaviorMilestoneView({{
        visualScene,
        evidenceFrame: {{ agent: {{ id: "robot", position: [1, 1, 0.5] }} }},
        trajectoryFrame: {{ objects: [{{ id: "robot", position: [1, 1, 0.5] }}] }},
      }});
      console.log(JSON.stringify({{ multi, agentOnly }}));
    """

    result = _run_node(root, script)

    assert result["multi"]["focus_ids"] == ["crate_a", "crate_b"]
    assert result["multi"]["distance"] > result["agentOnly"]["distance"]
    assert result["agentOnly"]["distance"] == pytest.approx(8.4)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_milestone_view_frames_both_sides_of_a_mechanism() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ buildBehaviorMilestoneView }} from {json.dumps(module_url)};
      const visualScene = {{
        objects: [
          {{ source_id: "robot", semantic_type: "agent", position: [3, 4, 0.5], size: [0.7, 0.7, 1] }},
          {{ source_id: "switch", position: [-3, 4, 0.1], size: [1, 1, 0.2] }},
          {{ source_id: "gate", position: [0, 2, 0.8], size: [2, 0.3, 1.6] }},
        ],
      }};
      const result = buildBehaviorMilestoneView({{
        visualScene,
        evidenceFrame: {{
          agent: {{ id: "robot", position: [-2.5, 4, 0.5] }},
          focus_ids: ["switch"],
        }},
        trajectoryFrame: {{}},
        mechanisms: [{{ id: "switch_opens_gate", trigger_id: "switch", gate_id: "gate" }}],
      }});
      console.log(JSON.stringify(result));
    """

    result = _run_node(root, script)

    assert result["focus_ids"] == ["switch", "gate"]
    assert result["distance"] > 8


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_nearest_trajectory_frame_is_deterministic() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ nearestTrajectoryFrame, nearestTrajectoryFrameIndex }} from {json.dumps(module_url)};
      const frames = [
        {{ total_step: 0, attempt: 1 }},
        {{ total_step: 10, attempt: 1 }},
        {{ total_step: 10, attempt: 2 }},
        {{ total_step: 20, attempt: 2 }},
      ];
      console.log(JSON.stringify({{
        near: nearestTrajectoryFrame(frames, 14).total_step,
        tie: nearestTrajectoryFrame(frames, 15).total_step,
        attempt: nearestTrajectoryFrame(frames, 10, 2).attempt,
        attemptFallback: nearestTrajectoryFrame(frames, 9, 3).attempt,
        laterAttemptIndex: nearestTrajectoryFrameIndex(frames, 10, 2),
        empty: nearestTrajectoryFrame([], 5),
        emptyIndex: nearestTrajectoryFrameIndex([], 5),
      }}));
    """

    result = _run_node(root, script)

    assert result == {
        "near": 10,
        "tie": 10,
        "attempt": 2,
        "attemptFallback": 1,
        "laterAttemptIndex": 2,
        "empty": None,
        "emptyIndex": -1,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_replay_frame_state_preserves_captured_tile_state() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ behaviorReplayFrameState }} from {json.dumps(module_url)};
      const frame = {{
        objects: [
          {{ id: "robot", position: [0, 0, 0.5], rotation_matrix: [1,0,0,0,1,0,0,0,1] }},
          {{ id: "crate", position: [2, 0, 0.5], rotation_matrix: [1,0,0,0,1,0,0,0,1] }},
        ],
        mechanisms: [{{ id: "gate", active: false }}],
      }};
      const fallback = behaviorReplayFrameState(frame, {{
        agent: {{ id: "robot", position: [1, 1, 0.5] }},
      }});
      const captured = behaviorReplayFrameState(frame, {{
        objects: [{{ id: "robot", position: [3, 4, 0.5] }}],
        mechanisms: [{{ id: "gate", active: true }}],
        view: {{ target: [3, 4, 0.5], distance: 8 }},
      }});
      console.log(JSON.stringify({{ fallback, captured }}));
    """

    result = _run_node(root, script)

    assert result["fallback"]["objects"][0]["position"] == [1, 1, 0.5]
    assert result["fallback"]["objects"][1]["position"] == [2, 0, 0.5]
    assert result["fallback"]["mechanisms"] == [{"id": "gate", "active": False}]
    assert result["captured"] == {
        "objects": [{"id": "robot", "position": [3, 4, 0.5]}],
        "mechanisms": [{"id": "gate", "active": True}],
        "view": {"target": [3, 4, 0.5], "distance": 8},
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_milestone_focus_stays_with_the_completed_check() -> None:
    root = Path(__file__).parents[1]
    module_url = (root / "environment_generation" / "studio_web" / "behavior_milestone_view.js").as_uri()
    script = f"""
      import {{ inheritBehaviorMilestoneFocus }} from {json.dumps(module_url)};
      const frames = inheritBehaviorMilestoneFocus([
        {{
          kind: "attempt",
          objective_focus: {{ check_id: "open_gate", target_ids: ["switch"] }},
          objective: {{ checks: [
            {{ id: "open_gate", description: "Open the gate", passed: false }},
            {{ id: "enter_goal", description: "Enter the goal", passed: false }},
          ] }},
        }},
        {{
          kind: "objective",
          label: "Completed: Open the gate",
          objective_focus: {{ check_id: "enter_goal", target_ids: ["goal"] }},
          objective: {{ checks: [
            {{ id: "open_gate", description: "Open the gate", passed: true }},
            {{ id: "enter_goal", description: "Enter the goal", passed: false }},
          ] }},
        }},
        {{
          kind: "final",
          label: "Completed: Enter the goal",
          objective_focus: {{ satisfied: true, target_ids: [] }},
          objective: {{ checks: [
            {{ id: "open_gate", description: "Open the gate", passed: true }},
            {{ id: "enter_goal", description: "Enter the goal", passed: true }},
          ] }},
        }},
      ]);
      console.log(JSON.stringify(frames.map((frame) => frame.focus_ids)));
    """

    result = _run_node(root, script)

    assert result == [["switch"], ["switch"], ["goal"]]
