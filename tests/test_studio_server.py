from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from environment_generation.artifacts import load_scene, persist_artifacts, persist_draft_spec
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.studio_server import (
    StudioConfig,
    behavior_regeneration_prompt,
    assistant_text_from_events,
    behavior_repair_prompt,
    build_codex_args,
    build_generation_prompt,
    build_revision_prompt,
    handle_behavior_run_request,
    handle_generate_request,
    handle_generate_stream_request,
    handle_revise_request,
    handle_revise_stream_request,
    handle_task_compile_request,
    handle_task_run_request,
    launch_play_session,
    normalize_history_turns,
    normalize_view_context,
    progress_update_for_event,
    courtyard_baseline_payload,
    delete_scene,
    prepare_generate_request,
    prepare_revise_request,
    read_history,
    recover_interrupted_task_compilations,
    slugify_env_id,
)
from environment_generation.studio_view_context import load_studio_view_context


def test_slugify_env_id() -> None:
    assert slugify_env_id("alex's 3d env") == "alex_s_3d_env"


def test_initial_generation_requires_an_explicit_environment_name(tmp_path) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)

    with pytest.raises(ValueError, match="environment must have a name"):
        prepare_generate_request(config, {"name": "   ", "prompt": "create a box"})

    assert list(tmp_path.iterdir()) == []


def test_public_branding_uses_environment_generation() -> None:
    studio_root = Path(__file__).parents[1] / "environment_generation" / "studio_web"
    html = (studio_root / "index.html").read_text(encoding="utf-8")
    server_source = (studio_root.parent / "studio_server.py").read_text(encoding="utf-8")

    assert "<title>Environment Generation</title>" in html
    assert "<strong>Environment Generation</strong>" in html
    assert "Use the environment-generation MCP tools" in server_source
    assert "Environment Generation running at" in server_source


def test_home_creation_actions_are_contextual_and_consistently_named() -> None:
    studio_root = Path(__file__).parents[1] / "environment_generation" / "studio_web"
    html = (studio_root / "index.html").read_text(encoding="utf-8")
    javascript = (studio_root / "studio.js").read_text(encoding="utf-8")
    stylesheet = (studio_root / "style.css").read_text(encoding="utf-8")

    assert 'id="newEnvButton" class="hidden"' in html
    assert 'id="homeCreateButton" class="hidden"' in html
    assert html.count("New Environment</button>") == 3
    assert 'els.newEnvButton.classList.toggle("hidden", page !== "env")' in javascript
    assert 'els.homeCreateButton.classList.toggle("hidden", !hasScenes)' in javascript
    assert 'els.emptyLibrary.classList.toggle("hidden", hasScenes)' in javascript
    assert 'els.topStatus.className = "status hidden"' in javascript
    assert "title.textContent = sceneDisplayName(scene)" in javascript
    assert "els.activeTitle.textContent = sceneDisplayName(scene)" in javascript
    assert 'id="primitiveBrowserButton" class="primitive-browser-button"' in html
    assert "Browse available objects and see how they look" in html
    assert 'openButton.className = "scene-card-open"' in javascript
    assert 'deleteButton.className = "scene-card-delete"' in javascript
    assert "async function deleteEnvironment(scene, button)" in javascript
    assert 'fetch(`/api/scenes/${encodeURIComponent(scene.env_id)}`, { method: "DELETE" })' in javascript
    assert "This permanently removes its tasks, tests, and saved runs." in javascript
    assert ".scene-card-open" in stylesheet
    assert ".scene-card-delete" in stylesheet


def test_delete_scene_removes_only_the_requested_environment(tmp_path) -> None:
    output_root = tmp_path / "generated"
    scene_dir = output_root / "demo"
    other_scene = output_root / "keep"
    (scene_dir / "tasks" / "task-1").mkdir(parents=True)
    (scene_dir / "tasks" / "task-1" / "report.json").write_text("{}", encoding="utf-8")
    other_scene.mkdir(parents=True)

    result = delete_scene(output_root, "demo")

    assert result == {"status": "success", "deleted": True, "env_id": "demo"}
    assert not scene_dir.exists()
    assert other_scene.is_dir()


def test_delete_scene_rejects_missing_or_unsafe_environment_ids(tmp_path) -> None:
    output_root = tmp_path / "generated"
    output_root.mkdir()

    with pytest.raises(FileNotFoundError):
        delete_scene(output_root, "missing")
    with pytest.raises(ValueError, match="invalid environment id"):
        delete_scene(output_root, ".")

    assert output_root.is_dir()


def test_create_page_uses_shell_only_threejs_preview() -> None:
    studio_root = Path(__file__).parents[1] / "environment_generation" / "studio_web"
    html = (studio_root / "index.html").read_text(encoding="utf-8")
    javascript = (studio_root / "studio.js").read_text(encoding="utf-8")
    stylesheet = (studio_root / "style.css").read_text(encoding="utf-8")
    primitive_catalog = (studio_root / "primitive_catalog.js").read_text(encoding="utf-8")
    visual_renderer = (studio_root / "visual_renderer.js").read_text(encoding="utf-8")
    ramp_geometry = (studio_root / "ramp_geometry.js").read_text(encoding="utf-8")
    server_source = (studio_root.parent / "studio_server.py").read_text(encoding="utf-8")

    assert 'id="createPreviewStage"' in html
    assert 'placeholder="Boxes and targets" required' in html
    assert (
        'placeholder="Place an agent in the bottom-right corner, a box in the middle, '
        'and a target region in the top-left."' in html
    )
    assert 'id="envNameError" class="form-field-error hidden"' in html
    assert 'els.envName.value = "studio_env"' not in javascript
    assert 'setEnvNameValidation("Environment must have a name.")' in javascript
    assert 'id="createOrbitSlider"' not in html
    assert 'id="createZoomOutButton"' not in html
    assert 'id="createZoomResetButton"' not in html
    assert 'id="createZoomInButton"' not in html
    assert 'id="createPreviewZoomLabel"' not in html
    assert 'id="createPrimitiveBrowserButton"' in html
    assert 'class="object-guide"' not in html
    assert 'id="activityState"' not in html
    assert 'id="activityLog"' not in html
    assert 'id="envActivityLog"' in html
    assert 'id="orbitSlider"' not in html
    assert 'id="zoomOutButton"' not in html
    assert 'id="zoomResetButton"' not in html
    assert 'id="zoomInButton"' not in html
    assert 'id="previewZoomLabel"' not in html
    assert "guide-agent" not in html
    assert "guide-crate" not in html
    assert "guide-goal" not in html
    assert 'fetch("/api/courtyard-baseline"' in javascript
    assert "captureCreateSubmission" in javascript
    assert "startCreatePreviewDrag" in javascript
    assert "handleCreatePreviewWheel" in javascript
    assert "dataset.cameraAzimuth" in javascript
    assert "structured_before_generation" in javascript
    assert "submitted_view_image: submittedView.imageDataUrl" in javascript
    assert 'id="primitiveBrowserButton"' in html
    assert 'id="modelPicker"' in html
    assert 'id="modelSelect"' in html
    assert 'fetch("/api/models"' in javascript
    assert "MODEL_STORAGE_KEY" in javascript
    assert 'populateCodexModelOptions(els.modelSelect, selected, "Codex default")' in javascript
    assert "${defaultLabel} (${defaultModelName})" in javascript
    assert "state.codexModels.defaultModel?.id" in javascript
    assert javascript.count("withCodexModel(") >= 8
    assert ".model-picker" in stylesheet
    assert 'if path == "/api/models"' in server_source
    assert 'role="dialog"' in html
    assert 'id="primitivePreviewStage"' in html
    assert 'id="playEventNotice"' in html
    assert "if (!els.playEventNotice.isConnected)" in javascript
    assert 'id="movementHelp" class="movement-help hidden"' in html
    assert "WASD" in html
    assert "Arrows" in html
    assert "Space" in html
    assert 'const controlMode = state.tasks.oracle.active ? "oracle" : state.play.active ? "play" : "";' in javascript
    assert "if (controlMode && !els.movementHelp.isConnected)" in javascript
    assert 'els.movementHelp.classList.toggle("hidden", !controlMode);' in javascript
    assert 'controlMode === "oracle" ? "Cancel" : "Exit"' in javascript
    assert ".movement-help" in stylesheet
    assert 'id="difficulty"' not in html
    assert "Physics Debug" not in html
    assert 'data-inspector-tab="files"' not in html
    assert 'data-inspector-pane="files"' not in html
    assert "Generated Files" not in html
    assert html.count('class="tab-info"') == 6
    assert 'data-tooltip="Runs deterministic checks against the scene specification and MuJoCo physics."' in html
    assert 'data-tooltip="Uses a vision-language model to review the styled scene from multiple camera angles."' in html
    assert 'data-tooltip="Runs autonomous attempts to check whether intended environment behaviors can be demonstrated."' in html
    assert 'data-tooltip="Contains benchmark objectives, deterministic trajectory tests, oracle solutions, and saved agent runs."' in html
    assert 'data-tooltip="Lists the authored physics objects and their properties."' in html
    assert 'data-tooltip="Shows recent generation steps, tool calls, and review progress."' in html
    assert 'id="inspectorTooltip" class="inspector-tooltip" role="tooltip" hidden' in html
    assert "showInspectorTooltip" in javascript
    assert "anchorRect.top - tooltipRect.height - gap" in javascript
    assert ".tab-info" in stylesheet
    assert ".inspector-tooltip" in stylesheet
    assert ".visually-hidden" in stylesheet
    assert "seed ${generation.seed}" not in javascript
    assert "openPrimitiveBrowser" in javascript
    assert 'els.createPrimitiveBrowserButton.addEventListener("click"' in javascript
    assert "setPrimitiveBrowserExpanded" in javascript
    assert "primitivePreviewScene" in javascript
    assert "const eyeGeometry = new THREE.SphereGeometry(radius * 0.13" in visual_renderer
    assert "new THREE.BoxGeometry(radius * 1.35, height * 0.2" not in visual_renderer
    assert "resizeRevisionPrompt" in javascript
    assert "node.dataset.rawContent" in javascript
    assert "Earlier updates" in javascript
    assert "appendConversationBlocks" in javascript
    assert "updateRevisionComposerState" in javascript
    assert 'classList.toggle("is-busy", busy)' in javascript
    assert 'setAttribute("aria-busy", busy ? "true" : "false")' in javascript
    assert 'aria-label="Describe the next environment change"' in html
    assert 'aria-label="Apply environment change"' in html
    assert ".composer button.is-busy" in stylesheet
    assert "min-width: 106px" in stylesheet
    assert "white-space: nowrap" in stylesheet
    assert 'const previewUrl = scene.styled_preview_url || "";' in javascript
    assert "scene.previews?.overview" not in javascript
    assert "Preparing styled preview" in javascript
    assert "replay verified" in javascript
    assert 'dataset.behaviorAction = "replay-frame"' in javascript
    assert "Number(button.dataset.replayStep || 0)" in javascript
    assert 'import { captureBehaviorMilestones } from "./behavior_milestone_capture.js";' in javascript
    assert 'from "./behavior_trial_view.js";' in javascript
    assert "buildBehaviorHeaderView" in javascript
    assert "Agent Tests" in html
    assert "progress-dots" in javascript
    assert "progress-dots" in stylesheet
    assert 'renderProgressDots(els.visualTabBadge, "Visual review running")' in javascript
    assert 'renderProgressDots(els.behaviorTabBadge, "Agent tests running")' in javascript
    assert "compact-during-run" not in javascript
    assert "compact-during-run" not in stylesheet
    assert 'renderProgressDots(pill, "Agent test running", "Running")' in javascript
    assert "runAllBehaviorTrials" in javascript
    assert "state.behavior.requests" in javascript
    assert "runView.retainedReport" in javascript
    assert "if (active && !trialActive) continue" not in javascript
    assert "Fallback locomotion" not in javascript
    assert "Immutable scene snapshot" not in javascript
    assert "Child agent running" not in javascript
    assert "Expected to demonstrate this affordance" not in javascript
    assert '"Passes when"' in javascript
    assert "Replay evidence" in javascript
    assert "First-person policy view" in javascript
    assert 'summary.textContent = "Run details"' in javascript
    for primitive_id in (
        "ground",
        "wall",
        "platform",
        "ramp",
        "static_box",
        "pushable_box",
        "ball",
        "cylinder",
        "agent",
        "goal",
        "target_region",
        "hazard",
        "floor_switch",
        "gate",
    ):
        assert f'primitive("{primitive_id}"' in primitive_catalog
    assert 'variant: "broken_paving"' in primitive_catalog
    assert 'createDangerZone(object, "spikes")' in visual_renderer
    assert 'import { rampRenderGeometry } from "./ramp_geometry.js";' in visual_renderer
    assert "export function rampRenderGeometry" in ramp_geometry
    assert '"/ramp_geometry.js"' in server_source
    assert '"/behavior_trial_view.js"' in server_source
    assert "addCautionFrame" in visual_renderer
    assert "addHazardWarningMark" in visual_renderer
    assert 'object.appearance?.variant === "stone"' not in visual_renderer
    assert 'primitive("platform"' in primitive_catalog and 'courtyard_platform", variant: "wood"' in primitive_catalog
    assert 'primitive("ramp"' in primitive_catalog and 'courtyard_ramp", variant: "wood"' in primitive_catalog
    assert 'object.semantic_type === "gate"' in visual_renderer
    assert "new THREE.OctahedronGeometry" in visual_renderer
    assert "packageMark" in visual_renderer
    assert "setCameraPose(pose)" in visual_renderer
    assert "setObjectVisibility(sourceId, visible)" in visual_renderer
    assert (studio_root / "styled_observation_capture.html").is_file()
    assert (studio_root / "styled_observation_capture.js").is_file()
    baseline = courtyard_baseline_payload()
    assert {obj["semantic_type"] for obj in baseline["objects"]} == {"ground", "wall"}
    assert not any(obj["semantic_type"] in {"agent", "goal", "hazard"} for obj in baseline["objects"])
    assert 'data-inspector-tab="tasks"' in html
    assert 'id="taskCreateForm"' in html
    assert 'id="taskOracleControls"' in html
    assert "Finish Recording" in html
    assert '"/task_recording.js"' in server_source
    assert '"/task_view.js"' in server_source
    assert "runValidatedTask" in javascript
    assert 'taskActionButton("Run Agent", "run", true)' in javascript
    assert "Run Codex" not in javascript
    assert "Running benchmark task" not in javascript
    assert 'renderProgressDots(els.taskTabBadge, "Task agent running")' in javascript
    assert 'renderProgressDots(statusNode, "Task agent running", "Running")' in javascript
    assert 'select.dataset.taskRunModel = ""' in javascript
    assert "resolvedTaskRunModel(modelSelection)" in javascript
    assert ".task-agent-controls" in stylesheet
    assert 'Accept: "text/event-stream"' in javascript
    assert "consumeTaskRunStream" in javascript
    assert 'from "./task_trajectory_replay.js";' in javascript
    assert "taskTrajectoryHistory" in javascript
    assert "startTaskTrajectoryReplay" in javascript
    assert 'identity.dataset.taskAction = "select-trajectory"' in javascript
    assert "loadTaskResultReport" in javascript
    assert "buildTaskResultView" in javascript
    assert "taskActivityAtStep" in javascript
    assert "Agent runs" in javascript
    assert 'id="taskRunOverlay"' in html
    assert 'id="taskRunOverlayTitle"' in html
    assert "task_scene_frame(payload)" in javascript
    assert "applyPhysicsState(frame.objects || [])" in javascript
    assert ".task-run-overlay" in stylesheet
    assert ".task-trajectory-list" in stylesheet
    assert ".task-run-live-frame" not in stylesheet
    assert 'emit=self._send_sse_event' in server_source
    assert '"/task_trajectory_replay.js"' in server_source
    assert "/oracle/start" in javascript
    assert 'id="visualReviewRepair"' in html
    assert "repairFromVisualReview" in javascript
    assert "/visual-reviews/${encodeURIComponent(reviewId)}/repair" in javascript


def test_handle_task_compile_request_persists_pending_oracle_task(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("task_demo", description="task")
    builder.add_ground_plane()
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_goal_zone(2, 0, id="goal")
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "task_demo", trace_records=[], render=False)

    def fake_compiler(*, scene_dir, instruction, spec, model="", repair_context=None):
        assert scene_dir == (tmp_path / "task_demo").resolve()
        assert instruction == "Reach the goal."
        assert spec["id"] == "task_demo"
        assert model == "test-model"
        assert repair_context is None
        return {
            "task_id": "reach_goal",
            "summary": "Reach the goal",
            "max_steps": 500,
            "tests": [
                {
                    "id": "reach",
                    "description": "Reach the goal.",
                    "mode": "all",
                    "ordered_condition_ids": [],
                    "conditions": [
                        {
                            "id": "enter",
                            "description": "Enter the goal.",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "robot"},
                                "target": {"id": "goal"},
                            },
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("environment_generation.studio_server.run_task_compiler", fake_compiler)
    result = handle_task_compile_request(
        config,
        env_id="task_demo",
        instruction="Reach the goal.",
        model="test-model",
    )

    assert result["status"] == "success"
    assert result["task"]["status"] == "pending_oracle"
    assert result["scene"]["task_summary"]["total"] == 1
    assert result["scene"]["tasks"][0]["oracle"] is None


def test_handle_task_compile_request_repairs_semantically_invalid_output(
    tmp_path, monkeypatch
) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("task_repair", description="task")
    builder.add_ground_plane()
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_goal_zone(2, 0, id="goal")
    persist_artifacts(
        spec=builder.finalize(),
        scene_dir=tmp_path / "task_repair",
        trace_records=[],
        render=False,
    )
    calls = []

    def fake_compiler(*, scene_dir, instruction, spec, model="", repair_context=None):
        calls.append(repair_context)
        temporal = "at_end" if repair_context is None else "eventually"
        if repair_context is not None:
            assert repair_context.attempt == 1
            assert "does not identify a chronological event" in repair_context.validation_errors[0]
            assert repair_context.rejected_output["tests"][0]["conditions"][0]["temporal"] == "at_end"
        return {
            "task_id": "reach_goal",
            "summary": "Reach the goal",
            "max_steps": 500,
            "tests": [
                {
                    "id": "reach",
                    "description": "Reach the goal.",
                    "mode": "all",
                    "ordered_condition_ids": ["enter"],
                    "conditions": [
                        {
                            "id": "enter",
                            "description": "Enter the goal.",
                            "temporal": temporal,
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "robot"},
                                "target": {"id": "goal"},
                            },
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("environment_generation.studio_server.run_task_compiler", fake_compiler)
    result = handle_task_compile_request(
        config,
        env_id="task_repair",
        instruction="Reach the goal.",
        model="test-model",
    )

    assert result["status"] == "success"
    assert len(calls) == 2
    assert calls[0] is None
    assert result["task"]["status"] == "pending_oracle"
    assert result["task"]["compiler"]["attempts"] == 2
    assert result["task"]["compiler"]["repair_attempts"] == 1
    assert len(result["task"]["compiler"]["validation_errors"]) == 1


def test_handle_task_compile_request_stops_after_two_failed_repairs(
    tmp_path, monkeypatch
) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("task_repair_limit", description="task")
    builder.add_ground_plane()
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_goal_zone(2, 0, id="goal")
    persist_artifacts(
        spec=builder.finalize(),
        scene_dir=tmp_path / "task_repair_limit",
        trace_records=[],
        render=False,
    )
    calls = []

    def fake_compiler(*, scene_dir, instruction, spec, model="", repair_context=None):
        calls.append(repair_context)
        return {
            "task_id": "reach_goal",
            "summary": "Reach the goal",
            "max_steps": 500,
            "tests": [
                {
                    "id": "reach",
                    "description": "Reach the goal.",
                    "mode": "all",
                    "ordered_condition_ids": ["enter"],
                    "conditions": [
                        {
                            "id": "enter",
                            "description": "Enter the goal.",
                            "temporal": "at_end",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "robot"},
                                "target": {"id": "goal"},
                            },
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("environment_generation.studio_server.run_task_compiler", fake_compiler)
    result = handle_task_compile_request(
        config,
        env_id="task_repair_limit",
        instruction="Reach the goal.",
    )

    assert result["status"] == "error"
    assert len(calls) == 3
    assert [context.attempt for context in calls[1:]] == [1, 2]
    assert result["task"]["status"] == "error"
    assert result["task"]["compiler"]["attempts"] == 3
    assert result["task"]["compiler"]["repair_attempts"] == 2
    assert len(result["task"]["compiler"]["validation_errors"]) == 3
    assert "does not identify a chronological event" in result["error"]


def test_handle_task_compile_request_does_not_retry_compiler_runtime_errors(
    tmp_path, monkeypatch
) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("task_runtime_error", description="task")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0, id="robot")
    persist_artifacts(
        spec=builder.finalize(),
        scene_dir=tmp_path / "task_runtime_error",
        trace_records=[],
        render=False,
    )
    calls = 0

    def fake_compiler(*, scene_dir, instruction, spec, model="", repair_context=None):
        nonlocal calls
        calls += 1
        raise RuntimeError("Codex process unavailable")

    monkeypatch.setattr("environment_generation.studio_server.run_task_compiler", fake_compiler)
    result = handle_task_compile_request(
        config,
        env_id="task_runtime_error",
        instruction="Move forward.",
    )

    assert result["status"] == "error"
    assert calls == 1
    assert result["task"]["compiler"]["attempts"] == 1
    assert result["task"]["compiler"]["validation_errors"] == []


def test_handle_task_run_request_returns_refreshed_scene(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("run_demo", description="task run")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "run_demo", trace_records=[], render=False)
    captured = {}

    def fake_runner(*, scene_dir, task_id, model=""):
        captured.update(scene_dir=scene_dir, task_id=task_id, model=model)
        return {"status": "success", "run_id": "run-0001-test", "report": {"passed": True}}

    monkeypatch.setattr("environment_generation.studio_server.run_validated_task", fake_runner)
    result = handle_task_run_request(
        config,
        env_id="run_demo",
        task_id="deliver",
        model="test-model",
    )

    assert result["report"]["passed"] is True
    assert result["scene"]["env_id"] == "run_demo"
    assert captured == {
        "scene_dir": (tmp_path / "run_demo").resolve(),
        "task_id": "deliver",
        "model": "test-model",
    }


def test_handle_task_run_request_forwards_live_events(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("stream_demo", description="task stream")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "stream_demo", trace_records=[], render=False)
    streamed: list[tuple[str, dict]] = []

    def fake_runner(*, scene_dir, task_id, model="", emit=None):
        assert emit is not None
        emit("text", {"delta": "Choosing a route."})
        return {"status": "success", "run_id": "run-0001-test", "report": {"passed": True}}

    monkeypatch.setattr("environment_generation.studio_server.run_validated_task", fake_runner)
    result = handle_task_run_request(
        config,
        env_id="stream_demo",
        task_id="deliver",
        model="test-model",
        emit=lambda event, data: streamed.append((event, data)),
    )

    assert result["report"]["passed"] is True
    assert streamed == [("text", {"delta": "Choosing a route."})]


def test_interrupted_task_compilation_becomes_retryable_error(tmp_path) -> None:
    from environment_generation.env_tasks import create_compiling_task, read_task

    builder = EnvSpec3DBuilder("compile_recovery", description="task")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    scene_dir = tmp_path / "compile_recovery"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    task = create_compiling_task(
        scene_dir=scene_dir,
        env_id="compile_recovery",
        instruction="Move one meter.",
    )

    recover_interrupted_task_compilations(scene_dir)

    recovered = read_task(scene_dir, task["task_id"], include_staleness=False)
    assert recovered["status"] == "error"
    assert "restarted" in recovered["compiler_error"]


def test_launch_play_session_starts_embedded_visual_session(tmp_path, monkeypatch) -> None:
    builder = EnvSpec3DBuilder("play_demo", description="play demo")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "play_demo"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    captured = {}

    class FakePlaySessions:
        def start(self, *, scene_dir):
            captured["scene_dir"] = scene_dir
            return {
                "session_id": "a" * 32,
                "env_id": "play_demo",
                "agent_id": "agent",
                "status": "Exploring",
                "objects": [],
            }

    monkeypatch.setattr("environment_generation.studio_server.PLAY_SESSIONS", FakePlaySessions())

    result = launch_play_session(env_id="play_demo", output_root=tmp_path)

    assert result["status"] == "success"
    assert result["mode"] == "embedded_visual"
    assert result["session_id"] == "a" * 32
    assert result["state"]["agent_id"] == "agent"
    assert captured["scene_dir"] == scene_dir.resolve()


def test_launch_play_session_rejects_unfinalized_scene(tmp_path) -> None:
    (tmp_path / "draft").mkdir()

    try:
        launch_play_session(env_id="draft", output_root=tmp_path)
    except ValueError as exc:
        assert "finalize" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("draft scene unexpectedly entered Play mode")


def test_launch_play_session_rejects_agentless_scene(tmp_path) -> None:
    builder = EnvSpec3DBuilder("static_scene", description="static")
    builder.add_wall(0, 0)
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "static_scene", trace_records=[], render=False)

    with pytest.raises(ValueError, match="playable agent"):
        launch_play_session(env_id="static_scene", output_root=tmp_path)


def test_generation_prompt_excludes_test_systems() -> None:
    prompt = build_generation_prompt(env_id="scene_1", user_prompt="make a room")

    assert "Apply this request to the saved blank courtyard" in prompt
    assert "resume_scene" in prompt
    assert "Do not call make_courtyard_level merely because this is the first turn" in prompt
    assert "add exactly the requested authored objects" in prompt
    assert "Never add a goal pad unless the request explicitly mentions" in prompt
    assert "Define deterministic env verification checks" in prompt
    assert "separate post-finalization multi-view visual review" in prompt
    assert "Do not define or run VLM/visual-review checks" in prompt
    assert "make exactly one current-turn agent-test decision" in prompt
    assert "Never let the generic fallback replace a prompt-specific request" in prompt
    assert "put a test that directly covers the latest request first" in prompt
    assert "game contract is required only when reach-goal gameplay has been explicitly authored" in prompt


def test_handle_generate_request_with_mocked_runner(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    captured = {}

    def fake_runner(*, prompt, env_id, output_root, model="", image_paths=()):
        baseline = load_scene(output_root / env_id)
        captured.update(prompt=prompt, baseline=baseline, image_paths=image_paths, model=model)
        return {
            "env_id": env_id,
            "events": [
                {"type": "tool_call", "name": "resume_scene", "message": "{}"},
                {"type": "assistant", "message": "done"},
            ],
            "stderr": [],
            "scene": baseline,
        }

    monkeypatch.setattr("environment_generation.studio_server.run_codex_generation", fake_runner)

    result = handle_generate_request(
        config,
        {"name": "Stacked Boxes", "prompt": "make a room", "model": "test-model"},
    )

    assert result["status"] == "success"
    assert result["env_id"] == "Stacked_Boxes"
    assert result["visual_review_id"].startswith("turn-0001-")
    assert result["scene"]["env_id"] == "Stacked_Boxes"
    assert result["scene"]["display_name"] == "Stacked Boxes"
    assert captured["baseline"]["display_name"] == "Stacked Boxes"
    assert captured["model"] == "test-model"
    assert {obj["semantic_type"] for obj in captured["baseline"]["objects"]} == {"ground", "wall"}
    assert result["scene"]["env_visual_review_pending"]["before"]["visual_scene_url"]
    assert "immutable blank-before baseline" in captured["prompt"]
    assert read_history(tmp_path, "Stacked_Boxes")[0] == {"role": "user", "content": "make a room"}
    assert read_history(tmp_path, "Stacked_Boxes")[1]["activity"] == [
        {"type": "tool_call", "label": "Tool: resume_scene", "message": "", "name": "resume_scene"}
    ]


def test_handle_behavior_run_request_returns_refreshed_scene(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("behavior_demo", description="behavior")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "behavior_demo", trace_records=[], render=False)
    captured = {}

    def fake_behavior_runner(*, scene_dir, model="", trial_ids=None, emit=None):
        captured.update(scene_dir=scene_dir, model=model, trial_ids=trial_ids)
        if emit:
            emit("behavior_trials", {"status": "running", "run_id": "run-0001-test"})
        return {
            "status": "success",
            "run_id": "run-0001-test",
            "report": {"status": "passed", "results": []},
        }

    monkeypatch.setattr("environment_generation.studio_server.run_behavior_trials", fake_behavior_runner)
    emitted = []

    result = handle_behavior_run_request(
        config,
        env_id="behavior_demo",
        model="test-model",
        trial_ids={"climb"},
        emit=lambda event, data: emitted.append((event, data)),
    )

    assert result["run_id"] == "run-0001-test"
    assert result["scene"]["capabilities"]["behavior_testable"] is True
    assert captured["scene_dir"] == (tmp_path / "behavior_demo").resolve()
    assert captured["trial_ids"] == {"climb"}
    assert emitted == [("behavior_trials", {"status": "running", "run_id": "run-0001-test"})]


def test_behavior_repair_prompt_uses_code_scored_failures(tmp_path) -> None:
    scene_dir = tmp_path / "repair_scene"
    scene_dir.mkdir()
    (scene_dir / "env_behavior_trials_report.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "trial_id": "climb",
                        "instruction": "Climb the tower.",
                        "status": "inconclusive",
                        "termination_reason": "step_budget",
                        "objective": {"checks": [{"id": "above", "passed": False}]},
                        "repair_hints": ["Review tower spacing."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prompt = behavior_repair_prompt(scene_dir)

    assert "code-scored behavior trial evidence" in prompt
    assert "Climb the tower" in prompt
    assert "step_budget" in prompt
    assert "inconclusive" in prompt


def test_behavior_regeneration_prompt_preserves_geometry_and_prior_intent(tmp_path) -> None:
    scene_dir = tmp_path / "stale_scene"
    scene_dir.mkdir()
    (scene_dir / "env_behavior_trials_plan.json").write_text(
        json.dumps({"prompt": "the central tower is too high to climb"}),
        encoding="utf-8",
    )

    prompt = behavior_regeneration_prompt(scene_dir)

    assert "Do not change scene objects, positions, dimensions, physics, or visuals" in prompt
    assert "latest behavior controller contract" in prompt
    assert "genuine attempt rules" in prompt
    assert "the central tower is too high to climb" in prompt


def test_handle_generate_stream_request_emits_workspace_progress_and_drafts(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)

    def fake_stream_runner(*, prompt, env_id, output_root, model="", on_event=None, image_paths=()):
        baseline = load_scene(output_root / env_id)
        builder = EnvSpec3DBuilder.from_spec_dict(baseline["spec"])
        persist_draft_spec(spec=builder.to_spec_dict(), scene_dir=output_root / env_id)
        if on_event:
            on_event({"type": "tool_call", "name": "resume_scene", "message": "{}"})
        box = builder.add_pushable_box(0, 0, id="requested_box")
        builder.set_object_appearance(box, "courtyard_pushable_crate")
        persist_draft_spec(spec=builder.to_spec_dict(), scene_dir=output_root / env_id)
        if on_event:
            on_event({"type": "tool_call", "name": "apply_operation", "message": "{}"})
            on_event({"type": "assistant", "message": "The box scene is taking shape."})
        persist_artifacts(spec=builder.finalize(), scene_dir=output_root / env_id, trace_records=[], render=False)
        if on_event:
            on_event({"type": "tool_call", "name": "finalize_scene", "message": "{}"})
        return {
            "env_id": env_id,
            "events": [{"type": "assistant", "message": "The box scene is taking shape."}],
            "stderr": [],
            "scene": None,
        }

    monkeypatch.setattr("environment_generation.studio_server.run_codex_generation_stream", fake_stream_runner)
    emitted = []

    handle_generate_stream_request(
        config,
        {"name": "live_demo", "prompt": "make a box scene"},
        lambda event, data: emitted.append((event, data)),
    )

    event_names = [event for event, _data in emitted]
    scene_updates = [data["scene"] for event, data in emitted if event == "scene"]
    assert event_names[0] == "system"
    assert emitted[0][1]["env_id"] == "live_demo"
    assert len(scene_updates) >= 2
    assert {obj["semantic_type"] for obj in scene_updates[0]["objects"]} == {"ground", "wall"}
    assert any(
        any(obj.get("semantic_type") == "pushable_box" for obj in scene.get("objects") or [])
        for scene in scene_updates[1:]
    )
    assert all(not scene.get("capabilities", {}).get("has_goal") for scene in scene_updates)
    assert ("text", {"delta": "The box scene is taking shape.\n"}) in emitted
    assert event_names[-1] == "done"
    assert emitted[-1][1]["status"] == "success"
    assert emitted[-1][1]["visual_review_id"].startswith("turn-0001-")
    loaded = load_scene(tmp_path / "live_demo")
    assert loaded["env_visual_review_pending"]["status"] == "awaiting_capture"
    assert loaded["env_visual_review_pending"]["before"]["visual_scene_url"]
    assert read_history(tmp_path, "live_demo")[-1]["content"] == "The box scene is taking shape."


def test_revision_prompt_uses_resume_scene() -> None:
    scene = {
        "env_id": "demo",
        "description": "box goal",
        "spec": {"world_size": [10, 8, 4], "gravity": [0, 0, -9.81], "theme": "storybook_adventure"},
        "objects": [{"id": "goal", "semantic_type": "goal", "position": [2, 0, 0.6]}],
        "cameras": [],
    }

    prompt = build_revision_prompt(
        env_id="demo",
        user_prompt="move the goal farther right",
        scene=scene,
        history=[{"role": "user", "content": "make a box goal scene"}],
        view_context={
            "capture_kind": "structured_before_edit",
            "preview_mode": "visual",
            "screen_space": {
                "source": "threejs_visual_camera",
                "azimuth_degrees": -42,
                "screen_right_world_xy": [0.743, 0.669],
                "screen_left_world_xy": [-0.743, -0.669],
                "reliable": True,
            },
            "visual": {
                "azimuth_degrees": -42,
                "screen_right_world_xy": [0.743, 0.669],
                "screen_left_world_xy": [-0.743, -0.669],
            },
        },
    )

    assert "resume_scene" in prompt
    assert "move the goal farther right" in prompt
    assert "do not create a new env id" in prompt
    assert "screen-space" in prompt
    assert "screen_right_world_xy" in prompt
    assert "structured_before_edit" in prompt
    assert "threejs_visual_camera" in prompt
    assert "record exactly one current-turn agent-test decision" in prompt
    assert "redefine the still-relevant prior tests against the current draft" in prompt
    assert "Choose the default test only when there is no concrete behavioral requirement" in prompt


def test_view_context_is_sanitized_for_prompt() -> None:
    value = normalize_view_context(
        {
            "preview_mode": "visual",
            "notes": "  left   right  ",
            "items": list(range(40)),
            "nested": {"ok": True, "bad": object()},
        }
    )

    assert value["preview_mode"] == "visual"
    assert value["notes"] == "left right"
    assert len(value["items"]) == 24
    assert value["nested"] == {"ok": True}


def test_revision_persists_and_attaches_exact_submitted_view(tmp_path) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("demo", description="demo")
    builder.make_box_goal_scene()
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "demo", trace_records=[], render=False)
    image = io.BytesIO()
    Image.new("RGB", (960, 540), (40, 120, 180)).save(image, format="PNG")
    image_data_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
    view_context = {
        "version": 2,
        "capture_kind": "structured_before_edit",
        "screen_space": {
            "source": "threejs_visual_camera",
            "camera": {
                "position": [3.4, 11.8, 9.5],
                "target": [0, 0, 0.5],
                "fov_y_degrees": 45,
                "aspect": 16 / 9,
            },
            "regions": {
                "bottom_left": {
                    "bounds_uv": {"left": 0.05, "top": 0.6, "right": 0.34, "bottom": 0.9},
                    "anchor": {"world_position": [-5, 5, 0], "screen_uv": [0.2, 0.75]},
                }
            },
        },
    }

    prepared = prepare_revise_request(
        config,
        "demo",
        {
            "message": "add an agent in the bottom left",
            "view_context": view_context,
            "submitted_view_image": image_data_url,
        },
    )

    assert len(prepared.image_paths) == 1
    assert prepared.image_paths[0].is_file()
    assert "exact PNG" in prepared.revision_prompt
    args = build_codex_args(
        prompt=prepared.revision_prompt,
        output_root=tmp_path,
        image_paths=prepared.image_paths,
    )
    assert args[args.index("--image") + 1] == str(prepared.image_paths[0])
    persisted = load_studio_view_context(tmp_path / "demo")
    assert persisted["requirements"][0]["region"] == "bottom_left"


def test_revision_evidence_is_attached_without_polluting_user_history(tmp_path) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("repair_demo", description="repair")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "repair_demo"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    evidence_path = scene_dir / "review_after.png"
    Image.new("RGB", (960, 540), (60, 110, 160)).save(evidence_path, format="PNG")

    prepared = prepare_revise_request(
        config,
        "repair_demo",
        {"message": "Fix the VLM review issues: the crate is hidden."},
        revision_evidence="Failed visual finding: reveal the requested crate.",
        evidence_image_paths=(evidence_path,),
    )

    assert prepared.image_paths == (evidence_path.resolve(),)
    assert "Studio-validated revision evidence follows" in prepared.revision_prompt
    assert "Failed visual finding: reveal the requested crate" in prepared.revision_prompt
    history = read_history(tmp_path, "repair_demo")
    assert history[-1] == {"role": "user", "content": "Fix the VLM review issues: the crate is hidden."}
    assert "Failed visual finding" not in history[-1]["content"]


def test_invalid_revision_evidence_is_rejected_before_creating_a_turn(tmp_path) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("repair_boundary", description="repair")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "repair_boundary"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    outside_path = tmp_path / "outside.png"
    Image.new("RGB", (960, 540), (60, 110, 160)).save(outside_path, format="PNG")

    with pytest.raises(ValueError, match="current environment"):
        prepare_revise_request(
            config,
            "repair_boundary",
            {"message": "Fix the visual review."},
            revision_evidence="Untrusted visual evidence.",
            evidence_image_paths=(outside_path,),
        )

    assert read_history(tmp_path, "repair_boundary") == []
    assert not (scene_dir / "visual_reviews").exists()


def test_initial_generation_persists_blank_before_view_and_spatial_context(tmp_path) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    image = io.BytesIO()
    Image.new("RGB", (960, 540), (50, 130, 190)).save(image, format="PNG")
    image_data_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
    view_context = {
        "version": 2,
        "capture_kind": "structured_before_generation",
        "screen_space": {
            "source": "threejs_visual_camera",
            "regions": {
                "bottom_left": {
                    "anchor": {"world_position": [-5, -4, 0], "screen_uv": [0.2, 0.8]},
                }
            },
        },
    }

    prepared = prepare_generate_request(
        config,
        {
            "name": "blank_first",
            "prompt": "put the robot in the bottom left",
            "seed": 31,
            "view_context": view_context,
            "submitted_view_image": image_data_url,
        },
    )
    baseline = load_scene(tmp_path / "blank_first")

    assert len(prepared.image_paths) == 1
    assert prepared.image_paths[0].is_file()
    assert {obj["semantic_type"] for obj in baseline["objects"]} == {"ground", "wall"}
    assert baseline["env_visual_review_pending"]["kind"] == "initial"
    assert baseline["env_visual_review_pending"]["before"]["visual_scene_url"]
    assert "structured_before_generation" in prepared.generation_prompt
    assert "bottom_left" in prepared.generation_prompt
    assert "exact PNG of the blank courtyard" in prepared.generation_prompt
    assert load_studio_view_context(tmp_path / "blank_first")["review_id"] == prepared.visual_review_id


def test_progress_update_filters_reasoning_and_formats_tools() -> None:
    assert progress_update_for_event({"type": "reasoning", "message": "hidden"}) is None
    assert progress_update_for_event({"type": "tool_call", "name": "resume_scene", "message": "{}"}) == {
        "type": "tool_call",
        "name": "resume_scene",
        "label": "Tool: resume_scene",
        "message": "",
    }


def test_agent_messages_are_public_assistant_text() -> None:
    assert assistant_text_from_events(
        [
            {"type": "reasoning", "message": "hidden"},
            {"type": "agent_message", "message": "I am placing the box."},
            {"type": "mcp_tool_call", "message": '{"env_id": "demo"}'},
        ],
        fallback="fallback",
    ) == "I am placing the box."


def test_long_public_agent_updates_are_not_cut_off_mid_sentence() -> None:
    updates = [f"Completed generation step {index}." for index in range(120)]
    result = assistant_text_from_events(
        [{"type": "agent_message", "message": update} for update in updates],
        fallback="fallback",
    )

    assert len(result) > 1600
    assert result.endswith("Completed generation step 119.")


def test_history_normalization_preserves_public_update_paragraphs() -> None:
    turns = normalize_history_turns(
        [{"role": "assistant", "content": "First complete update.\n\nSecond complete update."}]
    )

    assert turns == [{"role": "assistant", "content": "First complete update.\nSecond complete update."}]


def test_handle_revise_request_with_mocked_runner(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("demo", description="demo")
    builder.make_box_goal_scene()
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "demo", trace_records=[], render=False)
    captured = {}

    def fake_runner(*, prompt, env_id, output_root, model="", image_paths=()):
        captured.update(prompt=prompt, model=model)
        return {
            "env_id": env_id,
            "events": [{"type": "assistant", "message": "moved the goal"}],
            "stderr": [],
            "scene": None,
        }

    monkeypatch.setattr("environment_generation.studio_server.run_codex_generation", fake_runner)

    result = handle_revise_request(
        config,
        "demo",
        {
            "message": "move the goal",
            "model": "test-model",
            "view_context": {
                "preview_mode": "visual",
                "visual": {"screen_right_world_xy": [1, 0], "screen_left_world_xy": [-1, 0]},
            },
        },
    )

    assert result["status"] == "success"
    assert result["scene"]["env_id"] == "demo"
    assert result["visual_review_id"].startswith("turn-0001-")
    assert result["scene"]["env_visual_review_pending"]["kind"] == "revision"
    assert result["scene"]["env_visual_review_pending"]["before"]["visual_scene_url"]
    assert captured["model"] == "test-model"
    assert "resume_scene" in captured["prompt"]
    assert "screen_right_world_xy" in captured["prompt"]
    assert read_history(tmp_path, "demo")[-2:] == [
        {"role": "user", "content": "move the goal"},
        {"role": "assistant", "content": "moved the goal"},
    ]


def test_handle_revise_stream_request_emits_progress_and_done(tmp_path, monkeypatch) -> None:
    config = StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False)
    builder = EnvSpec3DBuilder("demo", description="demo")
    builder.make_box_goal_scene()
    persist_artifacts(spec=builder.finalize(), scene_dir=tmp_path / "demo", trace_records=[], render=False)

    def fake_stream_runner(*, prompt, env_id, output_root, model="", on_event=None, image_paths=()):
        assert "screen_right_world_xy" in prompt
        if on_event:
            on_event({"type": "reasoning", "message": "hidden"})
            on_event({"type": "tool_call", "name": "resume_scene", "message": "{}"})
            on_event({"type": "assistant", "message": "moved the goal"})
        return {
            "env_id": env_id,
            "events": [{"type": "assistant", "message": "moved the goal"}],
            "stderr": [],
            "scene": None,
        }

    monkeypatch.setattr("environment_generation.studio_server.run_codex_generation_stream", fake_stream_runner)
    emitted = []

    handle_revise_stream_request(
        config,
        "demo",
        {
            "message": "move the goal",
            "view_context": {
                "preview_mode": "visual",
                "visual": {"screen_right_world_xy": [1, 0], "screen_left_world_xy": [-1, 0]},
            },
        },
        lambda event, data: emitted.append((event, data)),
    )

    event_names = [event for event, _data in emitted]
    progress_labels = [data.get("label") for event, data in emitted if event == "progress"]
    done = emitted[-1][1]
    assert event_names[0] == "system"
    assert "Tool: resume_scene" in progress_labels
    assert "hidden" not in "\n".join(str(data) for _event, data in emitted)
    assert ("text", {"delta": "moved the goal\n"}) in emitted
    assert event_names[-1] == "done"
    assert done["status"] == "success"
    assert done["visual_review_id"].startswith("turn-0001-")
    assert done["scene"]["env_visual_review_pending"]["status"] == "awaiting_capture"
    assert read_history(tmp_path, "demo")[-2:] == [
        {"role": "user", "content": "move the goal"},
        {
            "role": "assistant",
            "content": "moved the goal",
            "activity": [
                {
                    "type": "progress",
                    "label": "Preparing revision",
                    "message": "Loaded scene context and conversation history.",
                },
                {"type": "tool_call", "label": "Tool: resume_scene", "message": "", "name": "resume_scene"},
                {"type": "progress", "label": "Refreshing scene", "message": "Loading updated artifacts and history."},
            ],
        },
    ]
