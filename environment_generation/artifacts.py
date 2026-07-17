from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env_behavior_trials import (
    ENV_BEHAVIOR_TRIAL_PLAN_FILENAME,
    ENV_BEHAVIOR_TRIAL_REPORT_FILENAME,
    behavior_trial_history,
    behavior_trial_summary,
    load_behavior_trial_plan,
    load_behavior_trial_report,
)
from .env_verification import (
    ENV_VERIFICATION_PLAN_FILENAME,
    ENV_VERIFICATION_REPORT_FILENAME,
    env_verification_summary,
    load_env_verification_plan,
    load_env_verification_report,
    spec_hash,
)
from .env_visual_review import (
    env_visual_review_history,
    env_visual_review_pending,
    env_visual_review_summary,
    load_current_styled_preview,
    load_env_visual_review_report,
    visual_scene_hash,
)
from .mujoco_compile import validate_mjcf_loads, write_mjcf
from .preview import PreviewError, render_orbit_previews, render_preview
from .schema import EnvSpec3D, env_spec_to_dict
from .studio_view_context import STUDIO_VIEW_CONTEXT_FILENAME
from .env_tasks import list_tasks, task_catalog_summary
from .visual_scene import VISUAL_SCENE_FILENAME, compile_visual_scene


ENV_SPEC_FILENAME = "env_spec_3d.json"
DRAFT_ENV_SPEC_FILENAME = "draft_env_spec_3d.json"
WORLD_XML_FILENAME = "world.xml"
METADATA_FILENAME = "metadata.json"
TRACE_FILENAME = "generation_trace.jsonl"
HISTORY_FILENAME = "studio_history.json"
PREVIEW_FILENAMES = {
    "overview": "preview_overview.png",
    "agent": "preview_agent.png",
    "goal": "preview_goal.png",
}
ORBIT_PREVIEW_COUNT = 16


def _scene_display_name(value: Any, *, fallback: str) -> str:
    name = str(value or "").strip()
    return name[:120] if name else fallback


@dataclass(frozen=True)
class ArtifactPaths:
    scene_dir: Path
    env_spec: Path
    world_xml: Path
    metadata: Path
    trace: Path
    visual_scene: Path
    previews: dict[str, Path]
    orbit_previews: list[Path]


def artifact_paths(scene_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        scene_dir=scene_dir,
        env_spec=scene_dir / ENV_SPEC_FILENAME,
        world_xml=scene_dir / WORLD_XML_FILENAME,
        metadata=scene_dir / METADATA_FILENAME,
        trace=scene_dir / TRACE_FILENAME,
        visual_scene=scene_dir / VISUAL_SCENE_FILENAME,
        previews={camera: scene_dir / filename for camera, filename in PREVIEW_FILENAMES.items()},
        orbit_previews=[scene_dir / _orbit_preview_filename(index) for index in range(ORBIT_PREVIEW_COUNT)],
    )


def persist_artifacts(
    *,
    spec: EnvSpec3D,
    scene_dir: Path,
    trace_records: list[dict[str, Any]],
    render: bool = True,
    display_name: str | None = None,
) -> dict[str, Any]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(scene_dir)
    existing_metadata = _read_optional_json(paths.metadata)
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}
    resolved_display_name = _scene_display_name(
        display_name if display_name is not None else existing_metadata.get("display_name"),
        fallback=spec.id,
    )
    spec_json = env_spec_to_dict(spec)
    _atomic_write_json(paths.env_spec, spec_json)
    _atomic_write_json(paths.visual_scene, compile_visual_scene(spec))
    write_mjcf(spec, paths.world_xml)
    _atomic_write_jsonl(paths.trace, trace_records)

    validation = validate_mjcf_loads(paths.world_xml)
    previews: dict[str, dict[str, Any]] = {}
    orbit_previews: dict[str, Any] = {"status": "not_rendered", "frames": []}
    if render:
        for camera, path in paths.previews.items():
            try:
                render_preview(paths.world_xml, path, camera=camera)
                previews[camera] = {"path": str(path), "status": "rendered"}
            except PreviewError as exc:
                previews[camera] = {"path": str(path), "status": "failed", "error": str(exc)}
        try:
            render_orbit_previews(paths.world_xml, paths.orbit_previews)
            orbit_previews = {
                "status": "rendered",
                "frames": [{"path": str(path), "index": index} for index, path in enumerate(paths.orbit_previews)],
            }
        except PreviewError as exc:
            orbit_previews = {
                "status": "failed",
                "error": str(exc),
                "frames": [{"path": str(path), "index": index} for index, path in enumerate(paths.orbit_previews)],
            }

    metadata = {
        "schema_version": "1.0",
        "env_id": spec.id,
        "display_name": resolved_display_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": ENV_SPEC_FILENAME,
        "derived_world": WORLD_XML_FILENAME,
        "validation": validation,
        "previews": previews,
        "orbit_previews": orbit_previews,
        "artifacts": {
            "env_spec": _file_record(paths.env_spec, "source"),
            "visual_scene": _file_record(paths.visual_scene, "derived_visual_scene"),
            "world_xml": _file_record(paths.world_xml, "derived_mjcf"),
            "trace": _file_record(paths.trace, "generation_trace"),
        },
    }
    for camera, path in paths.previews.items():
        if path.is_file():
            metadata["artifacts"][f"preview_{camera}"] = _file_record(path, "preview")
    for index, path in enumerate(paths.orbit_previews):
        if path.is_file():
            metadata["artifacts"][f"preview_orbit_{index:02d}"] = _file_record(path, "preview")
    env_plan = scene_dir / ENV_VERIFICATION_PLAN_FILENAME
    env_report = scene_dir / ENV_VERIFICATION_REPORT_FILENAME
    if env_plan.is_file():
        metadata["artifacts"]["env_verification_plan"] = _file_record(env_plan, "verification")
    if env_report.is_file():
        metadata["artifacts"]["env_verification_report"] = _file_record(env_report, "verification")
    behavior_plan = scene_dir / ENV_BEHAVIOR_TRIAL_PLAN_FILENAME
    behavior_report = scene_dir / ENV_BEHAVIOR_TRIAL_REPORT_FILENAME
    if behavior_plan.is_file():
        metadata["artifacts"]["env_behavior_trial_plan"] = _file_record(behavior_plan, "behavior_plan")
    if behavior_report.is_file():
        metadata["artifacts"]["env_behavior_trial_report"] = _file_record(behavior_report, "behavior_report")
    studio_view_context = scene_dir / STUDIO_VIEW_CONTEXT_FILENAME
    if studio_view_context.is_file():
        metadata["artifacts"]["studio_view_context"] = _file_record(studio_view_context, "submitted_view_context")
    _atomic_write_json(paths.metadata, metadata)
    return {
        "paths": {
            "env_spec": str(paths.env_spec),
            "visual_scene": str(paths.visual_scene),
            "world_xml": str(paths.world_xml),
            "metadata": str(paths.metadata),
            "trace": str(paths.trace),
            "previews": {camera: str(path) for camera, path in paths.previews.items()},
            "orbit_previews": [str(path) for path in paths.orbit_previews],
        },
        "metadata": metadata,
    }


def append_trace_record(trace_path: Path, record: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def persist_draft_spec(*, spec: dict[str, Any], scene_dir: Path) -> Path:
    """Publish the current builder state for live Studio previews."""
    path = scene_dir / DRAFT_ENV_SPEC_FILENAME
    _atomic_write_json(path, spec)
    return path


def refresh_visual_scene_artifact(scene_dir: Path) -> Path:
    """Rebuild the derived visual scene without touching physics or authored state."""
    spec = _read_optional_json(scene_dir / ENV_SPEC_FILENAME)
    if not isinstance(spec, dict):
        raise ValueError("finalized env_spec_3d.json is required to refresh the visual scene")
    visual_path = scene_dir / VISUAL_SCENE_FILENAME
    _atomic_write_json(visual_path, compile_visual_scene(spec))

    metadata_path = scene_dir / METADATA_FILENAME
    metadata = _read_optional_json(metadata_path)
    if isinstance(metadata, dict):
        artifacts = metadata.setdefault("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
            metadata["artifacts"] = artifacts
        artifacts["visual_scene"] = _file_record(visual_path, "derived_visual_scene")
        _atomic_write_json(metadata_path, metadata)
    return visual_path


def load_scene(scene_dir: Path) -> dict[str, Any] | None:
    spec_path = scene_dir / ENV_SPEC_FILENAME
    if not spec_path.is_file():
        return None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = _read_optional_json(scene_dir / METADATA_FILENAME) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    visual_scene_path = scene_dir / VISUAL_SCENE_FILENAME
    visual_scene = _read_optional_json(visual_scene_path)
    if visual_scene is None:
        try:
            visual_scene = compile_visual_scene(spec)
        except Exception:
            visual_scene = None
    current_hash = spec_hash(spec)
    current_visual_hash = visual_scene_hash(visual_scene) if isinstance(visual_scene, dict) else None
    styled_preview = load_current_styled_preview(
        scene_dir,
        current_spec_hash=current_hash,
        current_visual_scene_hash=current_visual_hash,
    )
    trace_records = _read_jsonl(scene_dir / TRACE_FILENAME)
    operation_count = _operation_count(trace_records)
    env_verification_plan = load_env_verification_plan(scene_dir)
    env_verification_report = load_env_verification_report(scene_dir)
    capabilities = _scene_capabilities(spec)
    tasks = list_tasks(scene_dir, current_spec=spec)
    return {
        "env_id": spec.get("id") or scene_dir.name,
        "display_name": _scene_display_name(
            metadata.get("display_name"),
            fallback=str(spec.get("id") or scene_dir.name),
        ),
        "description": spec.get("description") or "",
        "scene_dir": str(scene_dir),
        "spec": spec,
        "visual_scene": visual_scene,
        "spec_hash": current_hash,
        "visual_scene_hash": current_visual_hash,
        "visual_scene_url": (
            f"/generated/{scene_dir.name}/{VISUAL_SCENE_FILENAME}?v={visual_scene_path.stat().st_mtime_ns}"
            if visual_scene_path.is_file()
            else None
        ),
        "styled_preview": styled_preview,
        "styled_preview_url": styled_preview.get("url") if styled_preview else None,
        "metadata": metadata,
        "objects": spec.get("objects") or [],
        "cameras": spec.get("cameras") or [],
        "game": spec.get("game"),
        "generation": spec.get("generation"),
        "mechanisms": spec.get("mechanisms") or [],
        "capabilities": capabilities,
        "env_verification": env_verification_summary(
            scene_dir,
            draft_hash=current_hash,
            operation_count=operation_count,
        ),
        "env_verification_plan": env_verification_plan,
        "env_verification_report": env_verification_report,
        "env_behavior_trials": behavior_trial_summary(
            scene_dir,
            current_spec=spec,
            operation_count=operation_count,
        ),
        "env_behavior_trial_plan": load_behavior_trial_plan(scene_dir),
        "env_behavior_trial_report": load_behavior_trial_report(scene_dir),
        "env_behavior_trial_history": behavior_trial_history(scene_dir),
        "env_visual_review": env_visual_review_summary(
            scene_dir,
            current_spec=spec,
            current_visual_scene=visual_scene,
        ),
        "env_visual_review_report": load_env_visual_review_report(scene_dir),
        "env_visual_review_pending": env_visual_review_pending(scene_dir),
        "env_visual_review_history": env_visual_review_history(scene_dir),
        "tasks": tasks,
        "task_summary": task_catalog_summary(scene_dir, current_spec=spec),
        "history": _read_history(scene_dir),
        "history_url": _history_url(scene_dir),
        "previews": _preview_urls(scene_dir),
        "orbit_previews": _orbit_preview_urls(scene_dir),
        "status": "finalized" if metadata else "draft",
    }


def load_draft_scene(scene_dir: Path) -> dict[str, Any] | None:
    spec_path = scene_dir / DRAFT_ENV_SPEC_FILENAME
    spec = _read_optional_json(spec_path)
    if not isinstance(spec, dict):
        return None
    metadata = _read_optional_json(scene_dir / METADATA_FILENAME) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        visual_scene = compile_visual_scene(spec)
    except Exception:
        visual_scene = None
    current_hash = spec_hash(spec)
    current_visual_hash = visual_scene_hash(visual_scene) if isinstance(visual_scene, dict) else None
    styled_preview = load_current_styled_preview(
        scene_dir,
        current_spec_hash=current_hash,
        current_visual_scene_hash=current_visual_hash,
    )
    trace_records = _read_jsonl(scene_dir / TRACE_FILENAME)
    operation_count = _operation_count(trace_records)
    capabilities = _scene_capabilities(spec)
    tasks = list_tasks(scene_dir, current_spec=spec)
    return {
        "env_id": spec.get("id") or scene_dir.name,
        "display_name": _scene_display_name(
            metadata.get("display_name"),
            fallback=str(spec.get("id") or scene_dir.name),
        ),
        "description": spec.get("description") or "",
        "scene_dir": str(scene_dir),
        "spec": spec,
        "visual_scene": visual_scene,
        "spec_hash": current_hash,
        "visual_scene_hash": current_visual_hash,
        "visual_scene_url": None,
        "styled_preview": styled_preview,
        "styled_preview_url": styled_preview.get("url") if styled_preview else None,
        "metadata": metadata,
        "objects": spec.get("objects") or [],
        "cameras": spec.get("cameras") or [],
        "game": spec.get("game"),
        "generation": spec.get("generation"),
        "mechanisms": spec.get("mechanisms") or [],
        "capabilities": capabilities,
        "env_verification": env_verification_summary(
            scene_dir,
            draft_hash=current_hash,
            operation_count=operation_count,
        ),
        "env_verification_plan": load_env_verification_plan(scene_dir),
        "env_verification_report": load_env_verification_report(scene_dir),
        "env_behavior_trials": behavior_trial_summary(
            scene_dir,
            current_spec=spec,
            operation_count=operation_count,
        ),
        "env_behavior_trial_plan": load_behavior_trial_plan(scene_dir),
        "env_behavior_trial_report": load_behavior_trial_report(scene_dir),
        "env_behavior_trial_history": behavior_trial_history(scene_dir),
        "env_visual_review": env_visual_review_summary(
            scene_dir,
            current_spec=spec,
            current_visual_scene=visual_scene,
        ),
        "env_visual_review_report": load_env_visual_review_report(scene_dir),
        "env_visual_review_pending": env_visual_review_pending(scene_dir),
        "env_visual_review_history": env_visual_review_history(scene_dir),
        "tasks": tasks,
        "task_summary": task_catalog_summary(scene_dir, current_spec=spec),
        "history": _read_history(scene_dir),
        "history_url": _history_url(scene_dir),
        "previews": _preview_urls(scene_dir),
        "orbit_previews": _orbit_preview_urls(scene_dir),
        "status": "draft",
    }


def load_live_scene(scene_dir: Path) -> dict[str, Any] | None:
    final_path = scene_dir / ENV_SPEC_FILENAME
    draft_path = scene_dir / DRAFT_ENV_SPEC_FILENAME
    if draft_path.is_file() and (
        not final_path.is_file() or draft_path.stat().st_mtime_ns > final_path.stat().st_mtime_ns
    ):
        return load_draft_scene(scene_dir) or load_scene(scene_dir)
    return load_scene(scene_dir) or load_draft_scene(scene_dir)


def list_scenes(output_root: Path) -> list[dict[str, Any]]:
    if not output_root.is_dir():
        return []
    scenes = []
    for path in sorted(output_root.iterdir(), reverse=True):
        if not path.is_dir() or path.name.startswith("."):
            continue
        scene = load_live_scene(path)
        if scene:
            scenes.append(scene)
    return scenes


def _preview_urls(scene_dir: Path) -> dict[str, str]:
    urls = {}
    for camera, filename in PREVIEW_FILENAMES.items():
        path = scene_dir / filename
        if path.is_file():
            urls[camera] = f"/generated/{scene_dir.name}/{filename}?v={path.stat().st_mtime_ns}"
    return urls


def _scene_capabilities(spec: dict[str, Any]) -> dict[str, bool]:
    objects = spec.get("objects") if isinstance(spec.get("objects"), list) else []
    semantics = {
        str(obj.get("semantic_type") or "").lower()
        for obj in objects
        if isinstance(obj, dict)
    }
    has_agent = "agent" in semantics
    has_goal = "goal" in semantics
    game = spec.get("game") if isinstance(spec.get("game"), dict) else None
    return {
        "has_agent": has_agent,
        "has_goal": has_goal,
        "has_ground": "ground" in semantics,
        "playable": has_agent,
        "behavior_testable": has_agent,
        "task_testable": has_agent,
        "gameplay_ready": bool(game and has_agent and has_goal),
    }


def _orbit_preview_urls(scene_dir: Path) -> list[str]:
    urls = []
    for index in range(ORBIT_PREVIEW_COUNT):
        filename = _orbit_preview_filename(index)
        path = scene_dir / filename
        if path.is_file():
            urls.append(f"/generated/{scene_dir.name}/{filename}?v={path.stat().st_mtime_ns}")
    return urls


def _history_url(scene_dir: Path) -> str | None:
    path = scene_dir / HISTORY_FILENAME
    if not path.is_file():
        return None
    return f"/generated/{scene_dir.name}/{HISTORY_FILENAME}?v={path.stat().st_mtime_ns}"


def _read_history(scene_dir: Path) -> list[dict[str, Any]]:
    payload = _read_optional_json(scene_dir / HISTORY_FILENAME)
    if not isinstance(payload, dict):
        return []
    turns = payload.get("turns")
    if not isinstance(turns, list):
        return []
    normalized: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            normalized_turn: dict[str, Any] = {"role": role, "content": content}
            activity = _read_history_activity(turn.get("activity"))
            if role == "assistant" and activity:
                normalized_turn["activity"] = activity
            normalized.append(normalized_turn)
    return normalized


def _read_history_activity(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    activity: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        event_type = " ".join(str(item.get("type") or "progress").split())[:80]
        label = " ".join(str(item.get("label") or item.get("name") or event_type or "Progress").split())[:120]
        message = " ".join(str(item.get("message") or "").split())[:500]
        name = " ".join(str(item.get("name") or "").split())[:120]
        if "reasoning" in event_type.lower() or "reasoning" in label.lower():
            continue
        if not message and not name:
            continue
        if not label and not message:
            continue
        event = {"type": event_type or "progress", "label": label or "Progress", "message": message}
        if name:
            event["name"] = name
        activity.append(event)
    return activity


def _operation_count(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        if record.get("event") == "apply_operation"
        and isinstance(record.get("output"), dict)
        and record["output"].get("success") is True
    )


def _orbit_preview_filename(index: int) -> str:
    return f"preview_orbit_{index:02d}.png"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        "".join(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_optional_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _file_record(path: Path, role: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "role": role,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
