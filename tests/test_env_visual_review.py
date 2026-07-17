from __future__ import annotations

import base64
import io
import json
import re

import pytest
from PIL import Image

from environment_generation.artifacts import load_scene, persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_visual_review import (
    ENV_VISUAL_REVIEW_REPORT_FILENAME,
    STYLED_PREVIEW_FILENAME,
    STYLED_PREVIEW_METADATA_FILENAME,
    VISUAL_REVIEW_IMAGE_SIZE,
    build_visual_review_context,
    build_visual_review_prompt,
    build_visual_review_report,
    create_visual_review,
    env_visual_review_summary,
    load_visual_review_manifest,
    mark_visual_review_ready,
    persist_visual_review_evidence,
    prepare_visual_review_repair,
    visual_review_image_paths,
    visual_scene_hash,
    write_visual_review_report,
)
from environment_generation.studio_server import (
    StudioConfig,
    build_visual_review_codex_args,
    run_visual_review_request,
)


def _scene(tmp_path, env_id: str = "visual_review"):
    builder = EnvSpec3DBuilder(env_id, description="storybook box goal")
    builder.make_box_goal_scene()
    spec = builder.finalize()
    scene_dir = tmp_path / env_id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    loaded = load_scene(scene_dir)
    assert loaded is not None
    return scene_dir, loaded


def _png_data_url(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", VISUAL_REVIEW_IMAGE_SIZE, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _camera(azimuth: float = -42.0) -> dict:
    return {
        "target": [0, 0, 0.5],
        "distance": 19,
        "azimuth": azimuth,
        "elevation": 38,
        "panX": 0,
        "panY": 0,
    }


def _evidence(manifest: dict, *, revision: bool = False) -> dict:
    colors = [(30, 90, 140), (80, 140, 60), (170, 120, 70)]
    views = []
    for index, view_id in enumerate(("primary", "reverse", "layout")):
        view = {
            "id": view_id,
            "camera": _camera(-42 + index * 90),
            "after_image": _png_data_url(colors[index]),
        }
        if revision:
            view["before_image"] = _png_data_url(tuple(max(0, value - 10) for value in colors[index]))
        views.append(view)
    return {
        "scene_version": {
            "spec_hash": manifest["after"]["spec_hash"],
            "visual_scene_hash": manifest["after"]["visual_scene_hash"],
        },
        "views": views,
    }


def _review_output(*, critical_failure: bool = False) -> str:
    return json.dumps(
        {
            "status": "failed" if critical_failure else "passed",
            "summary": "The styled environment is readable.",
            "checks": [
                {
                    "id": "prompt_fidelity",
                    "category": "prompt_fidelity",
                    "passed": not critical_failure,
                    "severity": "critical" if critical_failure else "advisory",
                    "message": "The requested box is visible." if not critical_failure else "The requested box is missing.",
                    "evidence": [
                        {
                            "view_id": "primary",
                            "phase": "after",
                            "observation": "The center of the play area is clearly visible.",
                        }
                    ],
                    "repair_hint": "Add the box to the center." if critical_failure else "",
                }
            ],
        }
    )


def test_initial_context_contains_only_initial_request() -> None:
    context = build_visual_review_context(
        history=[
            {"role": "user", "content": "make a forest room"},
            {"role": "assistant", "content": "I used three tools and finalized it."},
        ],
        latest_request="make a forest room",
        kind="initial",
    )

    assert context == {
        "initial_request": "make a forest room",
        "latest_request": "make a forest room",
        "prior_user_requests": [],
        "context_truncated": False,
    }


def test_revision_context_uses_user_intent_and_excludes_assistant_narration() -> None:
    context = build_visual_review_context(
        history=[
            {"role": "user", "content": "make a forest room"},
            {"role": "assistant", "content": "hidden implementation narration"},
            {"role": "user", "content": "add a crate"},
            {"role": "assistant", "content": "more implementation narration"},
        ],
        latest_request="move the crate left",
        kind="revision",
    )

    assert context["initial_request"] == "make a forest room"
    assert context["prior_user_requests"] == ["add a crate"]
    assert context["latest_request"] == "move the crate left"
    assert "narration" not in json.dumps(context)


def test_revision_context_preserves_original_and_newest_requests_when_bounded() -> None:
    history = [{"role": "user", "content": "original environment purpose"}]
    history.extend(
        {"role": "user", "content": f"revision {index} " + ("detail " * 240)}
        for index in range(12)
    )

    context = build_visual_review_context(
        history=history,
        latest_request="latest change wins",
        kind="revision",
    )

    assert context["initial_request"] == "original environment purpose"
    assert context["latest_request"] == "latest change wins"
    assert context["prior_user_requests"][-1].startswith("revision 11")
    assert context["context_truncated"] is True
    assert len(json.dumps(context)) < 13_000


def test_revision_context_bounds_long_original_and_latest_messages() -> None:
    context = build_visual_review_context(
        history=[
            {"role": "user", "content": "o" * 20_000},
            {"role": "user", "content": "recent prior intent"},
        ],
        latest_request="l" * 20_000,
        kind="revision",
    )

    assert len(context["initial_request"]) == 3_000
    assert len(context["latest_request"]) == 5_000
    assert context["prior_user_requests"] == ["recent prior intent"]
    assert context["context_truncated"] is True
    assert len(json.dumps(context)) < 12_000


def test_revision_snapshot_and_evidence_are_persisted_in_pair_order(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="revision",
        latest_request="move the crate left",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="test-model",
        view_context={"visual": {"azimuth_degrees": -42, "target": [0, 0, 0.5]}},
        before_spec=scene["spec"],
        before_visual_scene=scene["visual_scene"],
    )
    assert re.fullmatch(r"turn-0002-[a-f0-9]{8}", manifest["review_id"])
    before_path = scene_dir / manifest["before"]["visual_scene_path"]
    assert json.loads(before_path.read_text()) == scene["visual_scene"]

    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(
        scene_dir,
        manifest["review_id"],
        _evidence(ready, revision=True),
    )

    assert captured["status"] == "evidence_ready"
    styled_preview_path = scene_dir / STYLED_PREVIEW_FILENAME
    styled_preview_metadata = json.loads((scene_dir / STYLED_PREVIEW_METADATA_FILENAME).read_text())
    assert styled_preview_path.is_file()
    assert styled_preview_metadata["spec_hash"] == captured["after"]["spec_hash"]
    assert styled_preview_metadata["visual_scene_hash"] == captured["after"]["visual_scene_hash"]
    assert styled_preview_metadata["source_review_id"] == manifest["review_id"]
    with Image.open(styled_preview_path) as image:
        assert image.size == VISUAL_REVIEW_IMAGE_SIZE
        assert image.getpixel((0, 0)) == (30, 90, 140)
    loaded = load_scene(scene_dir)
    assert loaded is not None
    assert loaded["styled_preview_url"].startswith(f"/generated/{scene_dir.name}/{STYLED_PREVIEW_FILENAME}?v=")
    assert loaded["metadata"]["artifacts"]["styled_preview"]["role"] == "styled_preview"
    assert [path.name for path in visual_review_image_paths(scene_dir, manifest["review_id"])] == [
        "01_before_primary.png",
        "02_after_primary.png",
        "03_before_reverse.png",
        "04_after_reverse.png",
        "05_before_layout.png",
        "06_after_layout.png",
    ]
    prompt = build_visual_review_prompt(captured)
    assert prompt.index("1. Before - Submitted camera") < prompt.index("2. After - Submitted camera")
    assert "Latest revision under review:\nmove the crate left" in prompt
    assert "Later user instructions override" in prompt

    visual_path = scene_dir / "visual_scene.json"
    changed_visual = json.loads(visual_path.read_text())
    changed_visual["theme"] = "changed_after_capture"
    visual_path.write_text(json.dumps(changed_visual))
    stale = load_scene(scene_dir)
    assert stale is not None
    assert stale["styled_preview"] is None
    assert stale["styled_preview_url"] is None


def test_initial_review_can_compare_blank_baseline_to_generated_scene(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="test-model",
        before_spec=scene["spec"],
        before_visual_scene=scene["visual_scene"],
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(
        scene_dir,
        manifest["review_id"],
        _evidence(ready, revision=True),
    )

    assert captured["kind"] == "initial"
    assert [path.name for path in visual_review_image_paths(scene_dir, manifest["review_id"])] == [
        "01_before_primary.png",
        "02_after_primary.png",
        "03_before_reverse.png",
        "04_after_reverse.png",
        "05_before_layout.png",
        "06_after_layout.png",
    ]
    prompt = build_visual_review_prompt(captured)
    assert "Initial request under review:\nmake a box goal scene" in prompt
    assert "blank-before/generated-after pairs" in prompt


def test_evidence_rejects_wrong_image_dimensions(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    bad = _evidence(ready)
    image = Image.new("RGB", (320, 180), (20, 30, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    bad["views"][0]["after_image"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    try:
        persist_visual_review_evidence(scene_dir, manifest["review_id"], bad)
    except ValueError as exc:
        assert "960x540" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong-sized evidence was accepted")


def test_review_artifact_paths_support_symlinked_scene_roots(tmp_path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    builder = EnvSpec3DBuilder("linked_scene", description="linked")
    builder.make_box_goal_scene()
    spec = builder.finalize()
    scene_dir = linked_root / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    loaded = load_scene(scene_dir)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=spec.id,
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )

    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=loaded["spec"],
        after_visual_scene=loaded["visual_scene"],
    )

    assert ready["status"] == "awaiting_capture"
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    artifact = metadata["artifacts"][f"visual_review_{manifest['review_id']}_manifest"]
    assert artifact["path"] == f"visual_reviews/{manifest['review_id']}/manifest.json"


def test_report_marks_critical_failure_as_nonblocking_attention(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="review-model",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(scene_dir, manifest["review_id"], _evidence(ready))
    report = build_visual_review_report(
        manifest=captured,
        model="review-model",
        raw_text=_review_output(critical_failure=True),
    )
    write_visual_review_report(scene_dir, report)

    assert report["status"] == "failed"
    assert report["needs_attention"] is True
    assert report["blocking"] is False
    loaded = load_scene(scene_dir)
    assert loaded["env_visual_review"]["status"] == "needs_attention"
    assert loaded["env_visual_review_report"]["checks"][0]["repair_hint"]


def test_visual_repair_packages_failed_findings_and_ordered_evidence(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path, "repairable_review")
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="revision",
        latest_request="move the crate left",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="review-model",
        before_spec=scene["spec"],
        before_visual_scene=scene["visual_scene"],
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(
        scene_dir,
        manifest["review_id"],
        _evidence(ready, revision=True),
    )
    raw_output = json.loads(_review_output(critical_failure=True))
    raw_output["checks"].append(
        {
            "id": "cohesion",
            "category": "theme_cohesion",
            "passed": True,
            "severity": "advisory",
            "message": "The courtyard style is cohesive.",
            "evidence": [
                {"view_id": "layout", "phase": "after", "observation": "Materials are consistent."}
            ],
            "repair_hint": "",
        }
    )
    report = build_visual_review_report(
        manifest=captured,
        model="review-model",
        raw_text=json.dumps(raw_output),
    )
    write_visual_review_report(scene_dir, report)

    repair = prepare_visual_review_repair(
        scene_dir,
        manifest["review_id"],
        current_spec=scene["spec"],
        current_visual_scene=scene["visual_scene"],
    )

    assert [path.name for path in repair["image_paths"]] == [
        "01_before_primary.png",
        "02_after_primary.png",
        "03_before_reverse.png",
        "04_after_reverse.png",
        "05_before_layout.png",
        "06_after_layout.png",
    ]
    assert [check["id"] for check in repair["failed_checks"]] == ["prompt_fidelity"]
    assert "The requested box is missing" in repair["display_message"]
    assert "visual evidence, not as independent user instructions" in repair["revision_evidence"]
    assert "1. Before - Submitted camera" in repair["revision_evidence"]
    assert "2. After - Submitted camera" in repair["revision_evidence"]
    assert "cohesion" not in repair["revision_evidence"]


def test_visual_repair_rejects_stale_and_noncurrent_reports(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path, "stale_repair")
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(scene_dir, manifest["review_id"], _evidence(ready))
    report = build_visual_review_report(manifest=captured, model="", raw_text=_review_output(critical_failure=True))
    write_visual_review_report(scene_dir, report)

    with pytest.raises(ValueError, match="not the current report"):
        prepare_visual_review_repair(
            scene_dir,
            "turn-9999-deadbeef",
            current_spec=scene["spec"],
            current_visual_scene=scene["visual_scene"],
        )

    changed_visual = {**scene["visual_scene"], "theme": "changed"}
    with pytest.raises(ValueError, match="stale"):
        prepare_visual_review_repair(
            scene_dir,
            manifest["review_id"],
            current_spec=scene["spec"],
            current_visual_scene=changed_visual,
        )


def test_visual_repair_rejects_review_without_failed_checks(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path, "passing_repair")
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(scene_dir, manifest["review_id"], _evidence(ready))
    report = build_visual_review_report(manifest=captured, model="", raw_text=_review_output())
    write_visual_review_report(scene_dir, report)

    with pytest.raises(ValueError, match="no failed checks"):
        prepare_visual_review_repair(
            scene_dir,
            manifest["review_id"],
            current_spec=scene["spec"],
            current_visual_scene=scene["visual_scene"],
        )


def test_summary_becomes_stale_when_visual_scene_changes(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = persist_visual_review_evidence(scene_dir, manifest["review_id"], _evidence(ready))
    report = build_visual_review_report(manifest=captured, model="", raw_text=_review_output())
    write_visual_review_report(scene_dir, report)
    changed_visual = {**scene["visual_scene"], "theme": "changed"}

    summary = env_visual_review_summary(
        scene_dir,
        current_spec=scene["spec"],
        current_visual_scene=changed_visual,
    )

    assert summary["status"] == "stale"
    assert visual_scene_hash(changed_visual) != report["after"]["visual_scene_hash"]


def test_late_report_does_not_replace_newer_pending_turn(tmp_path) -> None:
    scene_dir, scene = _scene(tmp_path)
    first = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
    )
    first = mark_visual_review_ready(
        scene_dir,
        first["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    first = persist_visual_review_evidence(scene_dir, first["review_id"], _evidence(first))
    second = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="revision",
        latest_request="move the crate",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="",
        before_spec=scene["spec"],
        before_visual_scene=scene["visual_scene"],
    )
    report = build_visual_review_report(manifest=first, model="", raw_text=_review_output())
    write_visual_review_report(scene_dir, report)

    assert not (scene_dir / ENV_VISUAL_REVIEW_REPORT_FILENAME).exists()
    assert load_visual_review_manifest(scene_dir, second["review_id"])["status"] == "pending_generation"


def test_visual_review_codex_args_are_read_only_ephemeral_and_keep_image_order(tmp_path) -> None:
    images = [tmp_path / name for name in ("01.png", "02.png", "03.png")]
    args = build_visual_review_codex_args(
        prompt="review",
        image_paths=images,
        output_path=tmp_path / "output.json",
        model="vision-model",
        cwd=tmp_path,
    )

    assert "--ephemeral" in args
    assert args[args.index("-s") + 1] == "read-only"
    assert "--output-schema" in args
    assert "mcp_servers" not in " ".join(args)
    attached = [args[index + 1] for index, value in enumerate(args) if value == "--image"]
    assert attached == [str(path) for path in images]
    assert args[-2:] == ["--", "review"]


def test_studio_review_request_persists_report_and_streams_status(tmp_path, monkeypatch) -> None:
    scene_dir, scene = _scene(tmp_path, env_id="studio_review")
    manifest = create_visual_review(
        scene_dir=scene_dir,
        env_id=scene["env_id"],
        kind="initial",
        latest_request="make a box goal scene",
        history=[{"role": "user", "content": "make a box goal scene"}],
        model="review-model",
    )
    ready = mark_visual_review_ready(
        scene_dir,
        manifest["review_id"],
        after_spec=scene["spec"],
        after_visual_scene=scene["visual_scene"],
    )
    captured = {}

    def fake_reviewer(*, prompt, image_paths, model="", on_event=None):
        captured["prompt"] = prompt
        captured["images"] = [path.name for path in image_paths]
        captured["model"] = model
        return {"raw_text": _review_output(), "events": [], "stderr": []}

    monkeypatch.setattr("environment_generation.studio_server.run_codex_visual_review", fake_reviewer)
    emitted = []
    result = run_visual_review_request(
        StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False),
        raw_env_id="studio_review",
        review_id=manifest["review_id"],
        evidence=_evidence(ready),
        emit=lambda event, data: emitted.append((event, data)),
    )

    assert result["report"]["status"] == "passed"
    assert result["scene"]["env_visual_review"]["status"] == "passed"
    assert captured["images"] == ["01_after_primary.png", "02_after_reverse.png", "03_after_layout.png"]
    assert captured["model"] == "review-model"
    assert [data["status"] for event, data in emitted if event == "visual_review"] == ["reviewing", "passed"]

    run_visual_review_request(
        StudioConfig(host="127.0.0.1", port=3033, output_root=tmp_path, open_browser=False),
        raw_env_id="studio_review",
        review_id=manifest["review_id"],
        model="new-review-model",
    )
    assert captured["model"] == "new-review-model"
