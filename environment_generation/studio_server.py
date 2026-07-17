from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .behavior_runner import dismiss_behavior_trials, recover_abandoned_behavior_runs, run_behavior_trials
from .artifacts import (
    ENV_SPEC_FILENAME,
    HISTORY_FILENAME,
    WORLD_XML_FILENAME,
    list_scenes,
    load_live_scene,
    load_scene,
    persist_artifacts,
)
from .courtyard import courtyard_shell_spec
from .codex_models import CODEX_MODEL_CATALOG
from .mcp_server import MCP_SERVER_NAME, WORKFLOW_GUIDE
from .runtime_config import rendering_subprocess_env, runtime_env_key
from .env_visual_review import (
    VISUAL_REVIEW_OUTPUT_SCHEMA_PATH,
    build_visual_review_error_report,
    build_visual_review_prompt,
    build_visual_review_report,
    create_visual_review,
    env_visual_review_summary,
    mark_visual_review_aborted,
    mark_visual_review_ready,
    mark_visual_review_reviewing,
    persist_visual_review_evidence,
    prepare_visual_review_repair,
    visual_review_image_paths,
    write_visual_review_report,
)
from .play_session import PlaySessionManager
from .env_tasks import (
    TaskDefinitionError,
    create_compiling_task,
    delete_task,
    finish_compiling_task,
    list_tasks,
    mark_task_compile_error,
    read_task,
)
from .task_compiler import (
    MAX_TASK_COMPILER_REPAIR_ATTEMPTS,
    TaskCompilerRepairContext,
    run_task_compiler,
)
from .task_oracle import TaskOracleSessionManager
from .task_runner import recover_abandoned_task_runs, run_validated_task
from .schema import env_spec_to_dict
from .studio_view_context import (
    persist_studio_view_context,
    submitted_view_image_path,
)
from .visual_scene import compile_visual_scene


SERVER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_ROOT.parent
WEB_ROOT = SERVER_ROOT / "studio_web"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generated"
RUNNER_TIMEOUT_SECONDS = 20 * 60
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "CREDENTIAL")
MAX_HISTORY_TURNS = 80
MAX_HISTORY_TURN_CHARS = 24000
MAX_PROMPT_HISTORY_TURNS = 20
MAX_PROMPT_HISTORY_CHARS = 12000
MAX_VIEW_CONTEXT_CHARS = 24000
MAX_ACTIVITY_EVENTS = 80
MAX_REQUEST_BODY_BYTES = 28 * 1024 * 1024
PLAY_SESSIONS = PlaySessionManager()
TASK_ORACLE_SESSIONS = TaskOracleSessionManager()
ACTIVE_TASK_COMPILATIONS: set[tuple[str, str]] = set()
TASK_COMPILATION_LOCK = threading.Lock()
ACTIVE_RUN_STATUSES = frozenset({"preparing", "running", "replaying", "evaluating"})
AGENT_TEST_DECISION_REMINDER = (
    "Before finalizing a scene with an agent, record exactly one current-turn agent-test decision. "
    "Define prompt-specific tests when the latest request introduces or changes an affordance, covering the "
    "latest requested behavior first and using a second test only for a relevant regression. When an edit can "
    "affect an existing affordance without adding a new one, redefine the still-relevant prior tests against the "
    "current draft. Preserve existing tests only for edits that cannot affect behavior or test semantics. Choose "
    "the default test only when there is no concrete behavioral requirement in the conversation."
)


@dataclass(frozen=True)
class StudioConfig:
    host: str
    port: int
    output_root: Path
    open_browser: bool


@dataclass(frozen=True)
class PreparedRevision:
    env_id: str
    prompt: str
    model: str
    revision_prompt: str
    visual_review_id: str
    image_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PreparedGeneration:
    env_id: str
    prompt: str
    model: str
    generation_prompt: str
    visual_review_id: str
    image_paths: tuple[Path, ...] = ()


def slugify_env_id(value: str, *, fallback: str = "studio_env") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


def next_env_id(output_root: Path, requested_name: str = "") -> str:
    base = slugify_env_id(requested_name) if requested_name.strip() else ""
    if base and not (output_root / base).exists():
        return base
    prefix = base or "studio_env"
    index = 1
    while True:
        candidate = f"{prefix}_{index:03d}" if base else f"studio_env_{index:03d}"
        if not (output_root / candidate).exists():
            return candidate
        index += 1


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def mcp_command(output_root: Path) -> tuple[str, list[str], dict[str, str]]:
    env = {
        **rendering_subprocess_env(),
        "PYTHONPATH": str(PROJECT_ROOT),
        runtime_env_key("OUTPUT_ROOT"): str(output_root),
    }
    return sys.executable, ["-m", "environment_generation.mcp_server"], env


def build_codex_args(
    *,
    prompt: str,
    output_root: Path,
    model: str = "",
    cwd: Path | None = None,
    image_paths: list[Path] | tuple[Path, ...] = (),
) -> list[str]:
    command, mcp_args, env = mcp_command(output_root)
    args = [
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--strict-config",
        "-s",
        "workspace-write",
        "-c",
        'approval_policy="never"',
    ]
    if cwd is not None:
        args.extend(["-C", str(cwd)])
    args.extend(
        [
            "-c",
            f"mcp_servers.{MCP_SERVER_NAME}.command={_toml_value(command)}",
            "-c",
            f"mcp_servers.{MCP_SERVER_NAME}.args={_toml_value(mcp_args)}",
            "-c",
            f"mcp_servers.{MCP_SERVER_NAME}.default_tools_approval_mode=\"approve\"",
        ]
    )
    for key, value in env.items():
        args.extend(["-c", f"mcp_servers.{MCP_SERVER_NAME}.env.{key}={_toml_value(value)}"])
    for path in image_paths:
        args.extend(["--image", str(path)])
    if model:
        args.extend(["-m", model])
    args.extend(["--", prompt])
    return args


def build_visual_review_codex_args(
    *,
    prompt: str,
    image_paths: list[Path],
    output_path: Path,
    model: str = "",
    cwd: Path | None = None,
) -> list[str]:
    args = [
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--strict-config",
        "-s",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--output-schema",
        str(VISUAL_REVIEW_OUTPUT_SCHEMA_PATH),
        "--output-last-message",
        str(output_path),
    ]
    if cwd is not None:
        args.extend(["-C", str(cwd)])
    for path in image_paths:
        args.extend(["--image", str(path)])
    if model:
        args.extend(["-m", model])
    args.extend(["--", prompt])
    return args


def build_generation_prompt(
    *,
    env_id: str,
    user_prompt: str,
    difficulty: str = "medium",
    seed: int | None = None,
    family: str = "",
    view_context: dict[str, Any] | None = None,
    has_view_image: bool = False,
) -> str:
    selected_seed = _level_seed(env_id, user_prompt) if seed is None else seed
    family_instruction = (
        f"Use the {family} family."
        if family
        else "Choose the best family from barrier_route, slalom, push_lane, elevation, switch_gate, or mixed based on the request."
    )
    image_context = (
        "An exact PNG of the blank courtyard from the user's submitted camera is attached. "
        "Use it together with the structured screen-space regions for spatial language.\n\n"
        if has_view_image
        else ""
    )
    return (
        f"Apply this request to the saved blank courtyard with id {env_id!r}:\n"
        f"{user_prompt.strip()}\n\n"
        f"{format_view_context_for_prompt(view_context)}\n\n"
        f"{image_context}"
        f"Start by calling resume_scene for {env_id!r}, then inspect_scene with include_spec=true. "
        "The saved scene is the immutable blank-before baseline and already has its ground and four walls. "
        "Treat the request as a targeted edit: add exactly the requested authored objects and no automatic route, lane, "
        "obstacles, agent, goal, hazard, or mechanism. Never add a goal pad unless the request explicitly mentions a goal, "
        "charging pad, target, or destination. Never add an agent unless the request explicitly mentions an agent or robot. "
        "Do not call make_courtyard_level merely because this is the first turn. "
        f"Only if the request explicitly asks for a complete generated course or level variation, call make_courtyard_level "
        f"with difficulty={difficulty!r}, seed={selected_seed}, and this family guidance: {family_instruction}\n\n"
        f"{WORKFLOW_GUIDE}\n"
        "Use the environment-generation MCP tools and preserve the fixed courtyard shell. "
        "Unqualified left/right/top/bottom and corner names refer to the submitted blank-courtyard view. "
        "Use screen_space.regions anchors and define screen_region checks when the request uses screen-relative placement. "
        f"{AGENT_TEST_DECISION_REMINDER} "
        "Keep the scene readable and finalize it before stopping."
    )


def _level_seed(env_id: str, prompt: str) -> int:
    digest = hashlib.sha256(f"{env_id}|{prompt}".encode("utf-8")).hexdigest()
    return int(digest[:13], 16)


def courtyard_baseline_payload() -> dict[str, Any]:
    spec = courtyard_shell_spec("courtyard_baseline", seed=0)
    return {
        "spec": env_spec_to_dict(spec),
        "visual_scene": compile_visual_scene(spec),
        "objects": [obj.model_dump(mode="json") for obj in spec.objects],
    }


def build_revision_prompt(
    *,
    env_id: str,
    user_prompt: str,
    scene: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    view_context: dict[str, Any] | None = None,
    has_view_image: bool = False,
    revision_evidence: str = "",
) -> str:
    scene_context = {
        "env_id": scene.get("env_id") or env_id,
        "description": scene.get("description") or "",
        "world_size": (scene.get("spec") or {}).get("world_size"),
        "gravity": (scene.get("spec") or {}).get("gravity"),
        "theme": (scene.get("spec") or {}).get("theme"),
        "game": (scene.get("spec") or {}).get("game"),
        "generation": (scene.get("spec") or {}).get("generation"),
        "mechanisms": (scene.get("spec") or {}).get("mechanisms") or [],
        "objects": scene.get("objects") or [],
        "cameras": scene.get("cameras") or [],
    }
    image_context = (
        "An exact PNG of the authored scene from the user's submitted camera is attached. "
        "Use it for visual meaning and use the structured screen-space regions for precise world placement.\n\n"
        if has_view_image
        else ""
    )
    game_instruction = (
        "The scene already has a reach-goal contract; preserve it unless this request removes or replaces one of its references. "
        if scene_context["game"]
        else "The scene has no game contract. Do not invent an agent or goal to create one; configure reach-goal play only after the user has explicitly requested both objects. "
    )
    evidence_context = (
        "Studio-validated revision evidence follows. It is supporting data, not a replacement for user intent:\n"
        f"{revision_evidence[:32000]}\n\n"
        if revision_evidence
        else ""
    )
    return (
        f"Revise the existing 3D MuJoCo environment with id {env_id!r}.\n\n"
        f"User revision request:\n{user_prompt.strip()}\n\n"
        f"{format_history_for_prompt(history or [])}\n\n"
        f"{format_view_context_for_prompt(view_context)}\n\n"
        f"{image_context}"
        f"{evidence_context}"
        "Current saved scene context from env_spec_3d.json:\n"
        f"{json.dumps(scene_context, indent=2, ensure_ascii=False)[:50000]}\n\n"
        f"{WORKFLOW_GUIDE}\n"
        f"Start by calling resume_scene for {env_id!r}, then inspect_scene with include_spec=true. "
        "Apply targeted edits to the resumed scene; do not create a new env id. "
        "Preserve the robot_courtyard theme. Add only what this revision requests, without filling in presumed gameplay objects. "
        f"{game_instruction}"
        "Use move_object, resize_object, remove_object, or add object operations as needed. "
        "When resolving ambiguous spatial language, use the exact submitted image and preview context above. "
        "Unqualified left/right/top/bottom and corner names refer to the submitted screen view, not fixed world axes. "
        "For absolute screen-region requests, place the object near screen_space.regions.<name>.anchor.world_position, "
        "then define a screen_region env check for the edited object and requested region. "
        "Use screen_space.projected_objects for object-relative references and a screen_relation check for phrases such as left/right of another object. "
        "Use legacy left/right vectors only when the richer camera and region context is absent. "
        "Explicit world-left/world-right still means -x/+x, and explicit up/above/down/below means z-axis movement. "
        "If an object reference is ambiguous, prefer selected_object_id when present, otherwise inspect object IDs and semantics before editing. "
        f"{AGENT_TEST_DECISION_REMINDER} "
        "Validate and finalize the same environment before stopping. "
        "When done, reply in 1-4 short sentences with what changed and any limitation."
    )


def scene_dir_for(output_root: Path, env_id: str) -> Path:
    output_root = output_root.resolve()
    scene_dir = (output_root / env_id).resolve()
    try:
        scene_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("environment id resolves outside output root") from exc
    return scene_dir


class SceneBusyError(RuntimeError):
    """Raised when deleting a scene would race active Studio work."""


def delete_scene(output_root: Path, env_id: str) -> dict[str, Any]:
    if not env_id or slugify_env_id(env_id) != env_id:
        raise ValueError("invalid environment id")
    scene_dir = scene_dir_for(output_root, env_id)
    if not scene_dir.is_dir():
        raise FileNotFoundError(env_id)

    resolved_scene = str(scene_dir.resolve())
    with TASK_COMPILATION_LOCK:
        compiling = any(active_scene == resolved_scene for active_scene, _task_id in ACTIVE_TASK_COMPILATIONS)
    if compiling:
        raise SceneBusyError("this environment is still generating task tests")
    if PLAY_SESSIONS.has_env(env_id):
        raise SceneBusyError("stop Play mode before deleting this environment")
    if TASK_ORACLE_SESSIONS.has_scene(scene_dir):
        raise SceneBusyError("finish or cancel the oracle recording before deleting this environment")
    if _scene_has_active_run(scene_dir):
        raise SceneBusyError("wait for the running agent test or task to finish before deleting this environment")

    shutil.rmtree(scene_dir)
    return {"status": "success", "deleted": True, "env_id": env_id}


def _scene_has_active_run(scene_dir: Path) -> bool:
    for manifest_path in scene_dir.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("status") not in ACTIVE_RUN_STATUSES:
            continue
        try:
            pid = int(manifest.get("pid") or 0)
        except (ValueError, TypeError):
            continue
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        return True
    return False


def history_path_for(output_root: Path, env_id: str) -> Path:
    return scene_dir_for(output_root, env_id) / HISTORY_FILENAME


def read_history(output_root: Path, env_id: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(history_path_for(output_root, env_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    return normalize_history_turns(payload.get("turns"))


def write_history(output_root: Path, env_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_history_turns(turns)[-MAX_HISTORY_TURNS:]
    path = history_path_for(output_root, env_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "env_id": env_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "turns": normalized,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return normalized


def append_history_turn(
    output_root: Path,
    env_id: str,
    role: str,
    content: str,
    *,
    activity: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    turns = read_history(output_root, env_id)
    turn: dict[str, Any] = {"role": role, "content": content}
    public_activity = normalize_activity_events(activity)
    if role == "assistant" and public_activity:
        turn["activity"] = public_activity
    turns.append(turn)
    return write_history(output_root, env_id, turns)


def normalize_history_turns(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    turns: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = _normalize_history_content(item.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        turn: dict[str, Any] = {"role": role, "content": _truncate_history_text(content)}
        activity = normalize_activity_events(item.get("activity"))
        if role == "assistant" and activity:
            turn["activity"] = activity
        turns.append(turn)
    return turns


def _normalize_history_content(value: Any) -> str:
    lines = [" ".join(line.split()) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _truncate_history_text(text: str) -> str:
    if len(text) <= MAX_HISTORY_TURN_CHARS:
        return text
    candidate = text[:MAX_HISTORY_TURN_CHARS]
    boundaries = [candidate.rfind(marker) for marker in ("\n", ". ", "! ", "? ")]
    boundary = max(boundaries)
    if boundary >= int(MAX_HISTORY_TURN_CHARS * 0.7):
        candidate = candidate[: boundary + 1]
    return candidate.rstrip() + "\n..."


def normalize_activity_events(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        event_type = " ".join(str(item.get("type") or "progress").split())[:80]
        name = " ".join(str(item.get("name") or "").split())[:120]
        label = " ".join(str(item.get("label") or name or event_type or "Progress").split())[:120]
        message = " ".join(str(item.get("message") or "").split())[:500]
        if "reasoning" in event_type.lower() or "reasoning" in label.lower():
            continue
        if not message and not name:
            continue
        if not label and not message:
            continue
        event = {"type": event_type or "progress", "label": label or "Progress", "message": message}
        if name:
            event["name"] = name
        if events and events[-1] == event:
            continue
        events.append(event)
        if len(events) >= MAX_ACTIVITY_EVENTS:
            break
    return events


def format_history_for_prompt(history: list[dict[str, Any]]) -> str:
    turns = normalize_history_turns(history)[-MAX_PROMPT_HISTORY_TURNS:]
    if not turns:
        return "No prior Studio conversation for this environment."
    lines = ["Prior Studio conversation for this environment:"]
    total = 0
    for turn in turns:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        line = f"- {prefix}: {turn['content']}"
        if total + len(line) > MAX_PROMPT_HISTORY_CHARS:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def format_view_context_for_prompt(view_context: dict[str, Any] | None) -> str:
    normalized = normalize_view_context(view_context)
    if not normalized:
        return (
            "No user preview view context was captured for this revision. "
            "Use the canonical world convention for ambiguous directions: left=-x, right=+x, forward=+y, back=-y, up=+z."
        )
    encoded = json.dumps(normalized, indent=2, ensure_ascii=False)
    if len(encoded) > MAX_VIEW_CONTEXT_CHARS:
        encoded = encoded[:MAX_VIEW_CONTEXT_CHARS] + "\n... truncated ..."
    return (
        "Structured before-edit preview context captured at the moment the user submitted this revision:\n"
        f"{encoded}\n"
        "Treat this camera state together with the saved scene context below as the authoritative view before any edit is applied. "
        "Spatial interpretation: unqualified left/right/top/bottom and corner names refer to this submitted screen-space view, "
        "not fixed world axes. screen_space.regions contains actionable world anchors and normalized screen bounds; "
        "screen_space.projected_objects identifies where existing objects appeared. Use those values before legacy direction vectors."
    )


def normalize_view_context(value: Any) -> dict[str, Any]:
    normalized = _compact_json_value(value, max_depth=8)
    return normalized if isinstance(normalized, dict) else {}


def _compact_json_value(value: Any, *, max_depth: int) -> Any:
    if max_depth < 0:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return " ".join(value.split())[:500]
    if isinstance(value, list):
        return [
            item
            for item in (_compact_json_value(item, max_depth=max_depth - 1) for item in value[:24])
            if item is not None
        ]
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for raw_key, raw_item in list(value.items())[:80]:
            key = str(raw_key).strip()[:80]
            if not key:
                continue
            item = _compact_json_value(raw_item, max_depth=max_depth - 1)
            if item is not None:
                compacted[key] = item
        return compacted
    return None


def assistant_text_from_events(events: list[dict[str, Any]], *, fallback: str) -> str:
    candidates: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "").lower()
        name = str(event.get("name") or "").strip()
        message = str(event.get("message") or "").strip()
        if "reasoning" in event_type:
            continue
        if not message or name:
            continue
        if event_type in {"agent_message", "assistant", "assistant_message", "message", "response_item"}:
            candidates.append(message)
    return clean_assistant_history_text("\n".join(candidates)) or fallback


def clean_assistant_history_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[stderr]"):
            continue
        if stripped.lower().startswith("artifacts are in "):
            continue
        lines.append(stripped)
    return _truncate_history_text("\n".join(lines).strip())


def with_scene_history(scene: dict[str, Any] | None, output_root: Path, env_id: str) -> dict[str, Any] | None:
    if scene is None:
        return None
    scene["history"] = read_history(output_root, env_id)
    path = history_path_for(output_root, env_id)
    scene["history_url"] = (
        f"/generated/{env_id}/{HISTORY_FILENAME}?v={path.stat().st_mtime_ns}"
        if path.is_file()
        else None
    )
    return scene


def request_wants_sse(payload: dict[str, Any], accept_header: str) -> bool:
    return bool(payload.get("stream")) or "text/event-stream" in accept_header.lower()


def studio_child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in SECRET_ENV_MARKERS)
    }
    if extra:
        env.update(extra)
    return env


def launch_play_session(*, env_id: str, output_root: Path) -> dict[str, Any]:
    safe_env_id = slugify_env_id(env_id)
    if not env_id or safe_env_id != env_id:
        raise ValueError("invalid environment id")

    root = output_root.resolve()
    scene_dir = (root / safe_env_id).resolve()
    try:
        scene_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("environment path escapes the generated scene directory") from exc

    world_path = scene_dir / WORLD_XML_FILENAME
    spec_path = scene_dir / ENV_SPEC_FILENAME
    if not world_path.is_file() or not spec_path.is_file():
        raise ValueError("finalize the environment before entering Play mode")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    objects = spec.get("objects") if isinstance(spec, dict) else None
    if not isinstance(objects, list) or not any(
        isinstance(item, dict) and item.get("semantic_type") == "agent" for item in objects
    ):
        raise ValueError("this environment does not contain a playable agent")

    state = PLAY_SESSIONS.start(scene_dir=scene_dir)
    return {
        "status": "success",
        "env_id": safe_env_id,
        "mode": "embedded_visual",
        "session_id": state["session_id"],
        "state": state,
        "message": "Interactive visual play mode started.",
    }


def handle_task_compile_request(
    config: StudioConfig,
    *,
    env_id: str,
    instruction: str,
    model: str = "",
) -> dict[str, Any]:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("task instruction is required")
    scene_dir = scene_dir_for(config.output_root, env_id)
    scene = load_scene(scene_dir)
    if scene is None or scene.get("status") != "finalized":
        raise ValueError("finalize the environment before creating a task")
    if not scene.get("capabilities", {}).get("task_testable"):
        raise ValueError("tasks require a controllable agent in the environment")
    task = create_compiling_task(
        scene_dir=scene_dir,
        env_id=env_id,
        instruction=instruction,
    )
    compile_key = (str(scene_dir.resolve()), str(task["task_id"]))
    with TASK_COMPILATION_LOCK:
        ACTIVE_TASK_COMPILATIONS.add(compile_key)
    compile_attempts = 0
    validation_errors: list[str] = []
    repair_context: TaskCompilerRepairContext | None = None
    try:
        max_attempts = 1 + MAX_TASK_COMPILER_REPAIR_ATTEMPTS
        while compile_attempts < max_attempts:
            compile_attempts += 1
            output = run_task_compiler(
                scene_dir=scene_dir,
                instruction=instruction,
                spec=scene["spec"],
                model=model,
                repair_context=repair_context,
            )
            try:
                task = finish_compiling_task(
                    scene_dir=scene_dir,
                    task_id=task["task_id"],
                    compiler_output=output,
                    model=model,
                    compile_attempts=compile_attempts,
                    repaired_validation_errors=tuple(validation_errors),
                )
                break
            except TaskDefinitionError as exc:
                validation_errors.append(str(exc))
                if compile_attempts >= max_attempts:
                    raise
                repair_context = TaskCompilerRepairContext(
                    attempt=compile_attempts,
                    rejected_output=output,
                    validation_errors=(str(exc),),
                )
    except Exception as exc:
        task = mark_task_compile_error(
            scene_dir,
            task["task_id"],
            str(exc),
            model=model,
            compile_attempts=max(1, compile_attempts),
            validation_errors=tuple(validation_errors),
        )
        return {
            "status": "error",
            "error": str(exc),
            "task": task,
            "scene": load_live_scene(scene_dir),
        }
    finally:
        with TASK_COMPILATION_LOCK:
            ACTIVE_TASK_COMPILATIONS.discard(compile_key)
    return {
        "status": "success",
        "task": task,
        "scene": load_live_scene(scene_dir),
    }


def recover_interrupted_task_compilations(scene_dir: Path) -> None:
    scene_dir = scene_dir.resolve()
    with TASK_COMPILATION_LOCK:
        active = {
            task_id
            for active_scene, task_id in ACTIVE_TASK_COMPILATIONS
            if active_scene == str(scene_dir)
        }
    for task in list_tasks(scene_dir):
        task_id = str(task.get("task_id") or "")
        if task.get("status") == "compiling" and task_id not in active:
            mark_task_compile_error(
                scene_dir,
                task_id,
                "Studio restarted before trajectory-test generation completed. Retry the task.",
            )


def handle_task_run_request(
    config: StudioConfig,
    *,
    env_id: str,
    task_id: str,
    model: str = "",
    emit: Any | None = None,
) -> dict[str, Any]:
    scene_dir = scene_dir_for(config.output_root, env_id)
    runner_args: dict[str, Any] = {
        "scene_dir": scene_dir,
        "task_id": task_id,
        "model": model,
    }
    if emit is not None:
        runner_args["emit"] = emit
    result = run_validated_task(**runner_args)
    result["scene"] = load_live_scene(scene_dir)
    return result


def run_codex_generation(
    *,
    prompt: str,
    env_id: str,
    output_root: Path,
    model: str = "",
    image_paths: list[Path] | tuple[Path, ...] = (),
) -> dict[str, Any]:
    return run_codex_generation_stream(
        prompt=prompt,
        env_id=env_id,
        output_root=output_root,
        model=model,
        image_paths=image_paths,
    )


def run_codex_generation_stream(
    *,
    prompt: str,
    env_id: str,
    output_root: Path,
    model: str = "",
    on_event: Any | None = None,
    image_paths: list[Path] | tuple[Path, ...] = (),
) -> dict[str, Any]:
    if shutil.which("codex") is None:
        raise RuntimeError("'codex' is not on PATH; install/login to the Codex CLI first")
    run_cwd = Path(tempfile.mkdtemp(prefix="environment-generation-studio-"))
    args = build_codex_args(
        prompt=prompt,
        output_root=output_root,
        model=model,
        cwd=PROJECT_ROOT,
        image_paths=image_paths,
    )
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    proc = subprocess.Popen(
        ["codex", *args],
        cwd=str(PROJECT_ROOT),
        env=studio_child_env({"PYTHONPATH": str(PROJECT_ROOT)}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: list[str] = []

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())

    threading.Thread(target=read_stderr, daemon=True).start()
    assert proc.stdout is not None
    for line in proc.stdout:
        if time.monotonic() - started > RUNNER_TIMEOUT_SECONDS:
            proc.terminate()
            raise TimeoutError("Codex generation exceeded timeout")
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            summary = {"type": "stdout", "name": "", "message": line}
        else:
            summary = summarize_codex_event(raw)
        events.append(summary)
        if on_event is not None:
            on_event(summary)
    code = proc.wait(timeout=10)
    if code != 0:
        raise RuntimeError("\n".join(stderr_lines[-20:]) or f"Codex exited with status {code}")
    scene = load_scene(output_root / env_id)
    return {
        "env_id": env_id,
        "events": events,
        "stderr": stderr_lines[-20:],
        "scene": scene,
        "run_cwd": str(run_cwd),
    }


def run_codex_visual_review(
    *,
    prompt: str,
    image_paths: list[Path],
    model: str = "",
    on_event: Any | None = None,
) -> dict[str, Any]:
    if shutil.which("codex") is None:
        raise RuntimeError("'codex' is not on PATH; install/login to the Codex CLI first")
    if not image_paths or not all(path.is_file() for path in image_paths):
        raise ValueError("visual review image evidence is incomplete")
    if not VISUAL_REVIEW_OUTPUT_SCHEMA_PATH.is_file():
        raise RuntimeError("visual review output schema is missing")

    events: list[dict[str, Any]] = []
    stderr_lines: list[str] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="environment-generation-visual-review-") as run_dir:
        output_path = Path(run_dir) / "last_message.json"
        args = build_visual_review_codex_args(
            prompt=prompt,
            image_paths=image_paths,
            output_path=output_path,
            model=model,
            cwd=PROJECT_ROOT,
        )
        proc = subprocess.Popen(
            ["codex", *args],
            cwd=str(PROJECT_ROOT),
            env=studio_child_env({"PYTHONPATH": str(PROJECT_ROOT)}),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def read_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())

        threading.Thread(target=read_stderr, daemon=True).start()
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.monotonic() - started > RUNNER_TIMEOUT_SECONDS:
                proc.terminate()
                raise TimeoutError("Codex visual review exceeded timeout")
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "stdout", "name": "", "message": line[:1000]}
            else:
                event = summarize_codex_event(raw)
            events.append(event)
            if on_event is not None:
                on_event(event)
        code = proc.wait(timeout=10)
        if code != 0:
            raise RuntimeError("\n".join(stderr_lines[-20:]) or f"Codex exited with status {code}")
        try:
            raw_text = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            raw_text = ""
    if not raw_text:
        raise RuntimeError("Codex visual review returned no final response")
    return {
        "raw_text": raw_text,
        "events": events,
        "stderr": stderr_lines[-20:],
    }


def run_visual_review_request(
    config: StudioConfig,
    *,
    raw_env_id: str,
    review_id: str,
    evidence: dict[str, Any] | None = None,
    model: str = "",
    emit: Any | None = None,
) -> dict[str, Any]:
    env_id = slugify_env_id(raw_env_id)
    if not env_id or env_id != raw_env_id:
        raise ValueError("invalid environment id")
    scene_dir = scene_dir_for(config.output_root, env_id)
    if evidence is not None:
        persist_visual_review_evidence(scene_dir, review_id, evidence)
    manifest = mark_visual_review_reviewing(scene_dir, review_id)
    current_scene = load_scene(scene_dir) or {}
    summary = env_visual_review_summary(
        scene_dir,
        current_spec=current_scene.get("spec"),
        current_visual_scene=current_scene.get("visual_scene"),
    )
    if emit is not None:
        emit(
            "visual_review",
            {
                "env_id": env_id,
                "review_id": review_id,
                "status": "reviewing",
                "summary": summary,
                "message": "Reviewing three styled camera views.",
            },
        )

    model = str(model or manifest.get("model") or "").strip()
    if model == "default":
        model = ""
    prompt = build_visual_review_prompt(manifest)
    image_paths = visual_review_image_paths(scene_dir, review_id)
    try:
        result = run_codex_visual_review(
            prompt=prompt,
            image_paths=image_paths,
            model=model,
        )
        report = build_visual_review_report(
            manifest=manifest,
            model=model,
            raw_text=str(result.get("raw_text") or ""),
        )
    except Exception as exc:
        report = build_visual_review_error_report(
            manifest=manifest,
            model=model,
            message=f"Codex visual review failed: {exc}",
        )
    write_visual_review_report(scene_dir, report)
    scene = load_scene(scene_dir)
    response = {
        "status": "success",
        "env_id": env_id,
        "review_id": review_id,
        "report": report,
        "summary": scene.get("env_visual_review") if scene else {},
        "scene": with_scene_history(scene, config.output_root, env_id),
    }
    if emit is not None:
        emit(
            "visual_review",
            {
                "env_id": env_id,
                "review_id": review_id,
                "status": report.get("status"),
                "summary": response["summary"],
                "report": report,
                "message": (report.get("summary") or {}).get("message", ""),
            },
        )
    return response


def summarize_codex_event(raw: dict[str, Any]) -> dict[str, Any]:
    event_type = str(raw.get("type") or raw.get("msg", {}).get("type") or "event")
    text = ""
    name = ""
    item = raw.get("item") if isinstance(raw.get("item"), dict) else {}
    item_type = str(item.get("type") or "")
    if "reasoning" in event_type.lower() or "reasoning" in item_type.lower():
        return {"type": event_type, "name": "", "message": ""}
    for key in ("message", "text", "content"):
        value = raw.get(key)
        if isinstance(value, str):
            text = value
            break
    if item:
        event_type = str(item.get("type") or event_type)
        name = str(item.get("name") or item.get("tool_name") or item.get("tool") or "")
        if not text:
            text = str(item.get("text") or item.get("arguments") or "")
    return {
        "type": event_type,
        "name": name,
        "message": text[:1000],
    }


def progress_update_for_event(event: dict[str, Any]) -> dict[str, str] | None:
    event_type = str(event.get("type") or "event").strip()
    if not event_type or "reasoning" in event_type.lower():
        return None
    name = str(event.get("name") or "").strip()
    message = str(event.get("message") or "").strip()
    if name:
        return {
            "type": event_type,
            "name": name,
            "label": f"Tool: {name}",
            "message": "",
        }
    if not message:
        return None
    if event_type in {"agent_message", "assistant", "assistant_message", "message", "response_item"} and message:
        return None
    return {
        "type": event_type,
        "name": "",
        "label": _human_event_label(event_type),
        "message": message[:500],
    }


def public_activity_from_events(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    activity: list[dict[str, Any]] = []
    for event in events:
        progress = progress_update_for_event(event)
        if progress:
            activity.append(progress)
    return normalize_activity_events(activity)


def _human_event_label(event_type: str) -> str:
    normalized = event_type.replace("_", " ").replace("-", " ").strip()
    if not normalized:
        return "Progress"
    return normalized[:1].upper() + normalized[1:]


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "EnvironmentGeneration/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path in {
            "/studio.js",
            "/style.css",
            "/visual_renderer.js",
            "/ramp_geometry.js",
            "/visual_review_capture.js",
            "/submitted_view_capture.js",
            "/behavior_milestone_capture.js",
            "/behavior_milestone_view.js",
            "/behavior_trial_view.js",
            "/task_recording.js",
            "/task_trajectory_replay.js",
            "/task_view.js",
            "/courtyard_assets.js",
            "/primitive_catalog.js",
        } or path.startswith("/vendor/") or path.startswith("/assets/"):
            mime = self._mime_type(path)
            asset_path = (WEB_ROOT / path.lstrip("/")).resolve()
            if WEB_ROOT.resolve() not in asset_path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(asset_path, mime)
            return
        if path == "/api/scenes":
            self._send_json({"scenes": list_scenes(self._config.output_root)})
            return
        if path == "/api/models":
            self._send_json(CODEX_MODEL_CATALOG.get())
            return
        if path == "/api/courtyard-baseline":
            self._send_json(courtyard_baseline_payload())
            return
        if match := re.fullmatch(r"/api/scenes/([^/]+)/tasks", path):
            env_id = match.group(1)
            scene_dir = scene_dir_for(self._config.output_root, env_id)
            recover_interrupted_task_compilations(scene_dir)
            TASK_ORACLE_SESSIONS.recover_scene(scene_dir)
            recover_abandoned_task_runs(scene_dir)
            scene = load_scene(scene_dir)
            if scene is None:
                self._send_json({"error": "scene not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"tasks": list_tasks(scene_dir, current_spec=scene["spec"])})
            return
        if match := re.fullmatch(r"/api/scenes/([^/]+)/tasks/([^/]+)", path):
            env_id, task_id = match.groups()
            try:
                scene_dir = scene_dir_for(self._config.output_root, env_id)
                recover_interrupted_task_compilations(scene_dir)
                TASK_ORACLE_SESSIONS.recover_scene(scene_dir)
                recover_abandoned_task_runs(scene_dir, task_id=task_id)
                task = read_task(scene_dir, task_id)
            except FileNotFoundError:
                self._send_json({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"task": task})
            return
        if match := re.fullmatch(r"/api/task-oracle/([a-f0-9]{32})", path):
            try:
                self._send_json({"state": TASK_ORACLE_SESSIONS.snapshot(match.group(1))})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        if match := re.fullmatch(r"/api/scenes/([^/]+)", path):
            env_id = match.group(1)
            scene_dir = scene_dir_for(self._config.output_root, env_id)
            recover_interrupted_task_compilations(scene_dir)
            TASK_ORACLE_SESSIONS.recover_scene(scene_dir)
            recover_abandoned_task_runs(scene_dir)
            scene = load_live_scene(scene_dir)
            if scene is None:
                self._send_json({"error": "scene not found"}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"scene": scene})
            return
        if path.startswith("/generated/"):
            self._serve_generated(path.removeprefix("/generated/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = unquote(urlparse(self.path).path)
        if match := re.fullmatch(r"/api/scenes/([^/]+)/tasks/([^/]+)", path):
            env_id, task_id = match.groups()
            try:
                result = delete_task(scene_dir_for(self._config.output_root, env_id), task_id)
            except FileNotFoundError:
                self._send_json({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        if match := re.fullmatch(r"/api/scenes/([^/]+)", path):
            try:
                result = delete_scene(self._config.output_root, match.group(1))
            except FileNotFoundError:
                self._send_json({"error": "environment not found"}, status=HTTPStatus.NOT_FOUND)
                return
            except SceneBusyError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/play":
                payload = self._read_json()
                result = launch_play_session(
                    env_id=str(payload.get("env_id") or ""),
                    output_root=self._config.output_root,
                )
            elif match := re.fullmatch(r"/api/play/([a-f0-9]{32})/(step|reset|stop)", path):
                payload = self._read_json()
                session_id, action = match.groups()
                if action == "step":
                    state = PLAY_SESSIONS.step(
                        session_id,
                        right=payload.get("right", 0.0),
                        forward=payload.get("forward", 0.0),
                        camera_azimuth=payload.get("camera_azimuth", 0.0),
                        jump=payload.get("jump", False),
                    )
                    result = {"status": "success", "state": state}
                elif action == "reset":
                    result = {"status": "success", "state": PLAY_SESSIONS.reset(session_id)}
                else:
                    result = {"status": "success", "stopped": PLAY_SESSIONS.stop(session_id)}
            elif match := re.fullmatch(r"/api/scenes/([^/]+)/tasks/compile", path):
                payload = self._read_json()
                result = handle_task_compile_request(
                    self._config,
                    env_id=match.group(1),
                    instruction=str(payload.get("instruction") or ""),
                    model=str(payload.get("model") or ""),
                )
            elif match := re.fullmatch(r"/api/scenes/([^/]+)/tasks/([^/]+)/oracle/start", path):
                env_id, task_id = match.groups()
                result = {
                    "status": "success",
                    "state": TASK_ORACLE_SESSIONS.start(
                        scene_dir=scene_dir_for(self._config.output_root, env_id),
                        task_id=task_id,
                    ),
                }
            elif match := re.fullmatch(r"/api/scenes/([^/]+)/tasks/([^/]+)/run", path):
                payload = self._read_json()
                env_id, task_id = match.groups()
                if request_wants_sse(payload, self.headers.get("Accept", "")):
                    self._send_sse_headers()
                    try:
                        response = handle_task_run_request(
                            self._config,
                            env_id=env_id,
                            task_id=task_id,
                            model=str(payload.get("model") or ""),
                            emit=self._send_sse_event,
                        )
                        self._send_sse_event("done", response)
                    except BrokenPipeError:
                        return
                    except Exception as exc:
                        self._send_sse_event("done", {"isError": True, "error": str(exc)})
                    return
                result = handle_task_run_request(
                    self._config,
                    env_id=env_id,
                    task_id=task_id,
                    model=str(payload.get("model") or ""),
                )
            elif match := re.fullmatch(
                r"/api/task-oracle/([a-f0-9]{32})/(step|reset|finish|cancel)",
                path,
            ):
                payload = self._read_json()
                session_id, action = match.groups()
                if action == "step":
                    result = {
                        "status": "success",
                        "state": TASK_ORACLE_SESSIONS.step(
                            session_id,
                            right=payload.get("right", 0.0),
                            forward=payload.get("forward", 0.0),
                            camera_azimuth=payload.get("camera_azimuth", 0.0),
                            jump=payload.get("jump", False),
                            frames=payload.get("frames", 4),
                            evaluate_report=payload.get("evaluate_report", False),
                        ),
                    }
                elif action == "reset":
                    result = {"status": "success", "state": TASK_ORACLE_SESSIONS.reset(session_id)}
                elif action == "finish":
                    result = TASK_ORACLE_SESSIONS.finish(session_id)
                    task_result = result.get("task") if isinstance(result, dict) else None
                    if isinstance(task_result, dict):
                        result["scene"] = load_live_scene(
                            self._config.output_root / str(task_result.get("env_id") or "")
                        )
                else:
                    result = TASK_ORACLE_SESSIONS.cancel(session_id)
            elif path == "/api/generate":
                payload = self._read_json()
                if request_wants_sse(payload, self.headers.get("Accept", "")):
                    self._send_sse_headers()
                    try:
                        handle_generate_stream_request(self._config, payload, self._send_sse_event)
                    except BrokenPipeError:
                        return
                    except Exception as exc:
                        self._send_sse_event("text", {"delta": f"I got stuck before finishing the environment: {exc}"})
                        self._send_sse_event("done", {"isError": True, "error": str(exc)})
                    return
                result = handle_generate_request(self._config, payload)
            elif match := re.fullmatch(
                r"/api/scenes/([^/]+)/behavior-trials/(run|rerun|dismiss|repair|regenerate)",
                path,
            ):
                env_id, action = match.groups()
                payload = self._read_json()
                if action in {"run", "rerun"}:
                    selected = {
                        str(value)
                        for value in payload.get("trial_ids") or []
                        if value is not None
                    }
                    if request_wants_sse(payload, self.headers.get("Accept", "")):
                        self._send_sse_headers()
                        try:
                            response = handle_behavior_run_request(
                                self._config,
                                env_id=env_id,
                                model=str(payload.get("model") or ""),
                                trial_ids=selected,
                                emit=self._send_sse_event,
                            )
                            self._send_sse_event("done", response)
                        except BrokenPipeError:
                            return
                        except Exception as exc:
                            self._send_sse_event("done", {"isError": True, "error": str(exc)})
                        return
                    result = handle_behavior_run_request(
                        self._config,
                        env_id=env_id,
                        model=str(payload.get("model") or ""),
                        trial_ids=selected,
                    )
                elif action == "dismiss":
                    result = dismiss_behavior_trials(
                        scene_dir=scene_dir_for(self._config.output_root, env_id),
                        trial_ids={str(value) for value in payload.get("trial_ids") or [] if value is not None},
                    )
                    result["scene"] = load_live_scene(self._config.output_root / env_id)
                else:
                    revision_prompt = (
                        behavior_regeneration_prompt(scene_dir_for(self._config.output_root, env_id))
                        if action == "regenerate"
                        else behavior_repair_prompt(scene_dir_for(self._config.output_root, env_id))
                    )
                    revision_payload = {
                        "message": revision_prompt,
                        "model": str(payload.get("model") or ""),
                        "history": payload.get("history") or [],
                        "view_context": payload.get("view_context") or {},
                        "stream": payload.get("stream", False),
                    }
                    if request_wants_sse(revision_payload, self.headers.get("Accept", "")):
                        self._send_sse_headers()
                        try:
                            handle_revise_stream_request(
                                self._config,
                                env_id,
                                revision_payload,
                                self._send_sse_event,
                            )
                        except BrokenPipeError:
                            return
                        except Exception as exc:
                            self._send_sse_event("done", {"isError": True, "error": str(exc)})
                        return
                    result = handle_revise_request(self._config, env_id, revision_payload)
            elif match := re.fullmatch(
                r"/api/scenes/([^/]+)/visual-reviews/(turn-\d{4}-[a-f0-9]{8})/(evidence|rerun|repair)",
                path,
            ):
                env_id, review_id, action = match.groups()
                payload = self._read_json()
                if action == "repair":
                    scene_dir = scene_dir_for(self._config.output_root, env_id)
                    scene = load_scene(scene_dir)
                    if scene is None:
                        raise ValueError(f"scene {env_id!r} not found")
                    repair = prepare_visual_review_repair(
                        scene_dir,
                        review_id,
                        current_spec=scene.get("spec"),
                        current_visual_scene=scene.get("visual_scene"),
                    )
                    revision_payload = {
                        "message": repair["display_message"],
                        "model": str(payload.get("model") or ""),
                        "history": payload.get("history") or [],
                        "view_context": payload.get("view_context") or {},
                        "submitted_view_image": str(payload.get("submitted_view_image") or ""),
                        "stream": payload.get("stream", False),
                    }
                    if request_wants_sse(revision_payload, self.headers.get("Accept", "")):
                        self._send_sse_headers()
                        try:
                            handle_revise_stream_request(
                                self._config,
                                env_id,
                                revision_payload,
                                self._send_sse_event,
                                revision_evidence=str(repair["revision_evidence"]),
                                evidence_image_paths=repair["image_paths"],
                            )
                        except BrokenPipeError:
                            return
                        except Exception as exc:
                            self._send_sse_event("done", {"isError": True, "error": str(exc)})
                        return
                    result = handle_revise_request(
                        self._config,
                        env_id,
                        revision_payload,
                        revision_evidence=str(repair["revision_evidence"]),
                        evidence_image_paths=repair["image_paths"],
                    )
                else:
                    requested_model = str(payload.get("model") or "").strip()
                    evidence = (
                        {key: value for key, value in payload.items() if key not in {"model", "stream"}}
                        if action == "evidence"
                        else None
                    )
                    if request_wants_sse(payload, self.headers.get("Accept", "")) or "text/event-stream" in self.headers.get("Accept", "").lower():
                        self._send_sse_headers()
                        try:
                            response = run_visual_review_request(
                                self._config,
                                raw_env_id=env_id,
                                review_id=review_id,
                                evidence=evidence,
                                model=requested_model,
                                emit=self._send_sse_event,
                            )
                            self._send_sse_event("done", response)
                        except BrokenPipeError:
                            return
                        except Exception as exc:
                            self._send_sse_event("done", {"isError": True, "error": str(exc)})
                        return
                    result = run_visual_review_request(
                        self._config,
                        raw_env_id=env_id,
                        review_id=review_id,
                        evidence=evidence,
                        model=requested_model,
                    )
            else:
                match = re.fullmatch(r"/api/scenes/([^/]+)/revise", path)
                if not match:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json()
                if request_wants_sse(payload, self.headers.get("Accept", "")):
                    self._send_sse_headers()
                    try:
                        handle_revise_stream_request(self._config, match.group(1), payload, self._send_sse_event)
                    except BrokenPipeError:
                        return
                    except Exception as exc:
                        self._send_sse_event("text", {"delta": f"I got stuck before finishing the revision: {exc}"})
                        self._send_sse_event("done", {"isError": True, "error": str(exc)})
                    return
                result = handle_revise_request(self._config, match.group(1), payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    @property
    def _config(self) -> StudioConfig:
        return self.server.studio_config  # type: ignore[attr-defined]

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            raise ValueError("request body exceeds the size limit")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(self, value: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_sse_event(self, event: str, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        for line in payload.splitlines() or [""]:
            self.wfile.write(f"data: {line}\n".encode("utf-8"))
        self.wfile.write(b"\n")
        self.wfile.flush()
        if event == "done":
            self.close_connection = True

    @staticmethod
    def _mime_type(path: str) -> str:
        if path.endswith(".js"):
            return "text/javascript; charset=utf-8"
        if path.endswith(".css"):
            return "text/css; charset=utf-8"
        if path.endswith(".json"):
            return "application/json; charset=utf-8"
        if path.endswith(".glb"):
            return "model/gltf-binary"
        if path.endswith(".gltf"):
            return "model/gltf+json"
        return "application/octet-stream"

    def _serve_file(self, path: Path, mime: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_generated(self, relative: str) -> None:
        safe_parts = [part for part in Path(relative).parts if part not in {"..", "/", ""}]
        path = self._config.output_root.joinpath(*safe_parts)
        if not path.is_file() or self._config.output_root not in path.resolve().parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = "image/png" if path.suffix == ".png" else "application/octet-stream"
        if path.suffix == ".json":
            mime = "application/json; charset=utf-8"
        elif path.suffix == ".xml":
            mime = "application/xml; charset=utf-8"
        elif path.suffix == ".jsonl":
            mime = "application/jsonl; charset=utf-8"
        self._serve_file(path, mime)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))


def handle_generate_request(config: StudioConfig, payload: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_generate_request(config, payload)
    try:
        result = run_codex_generation(
            prompt=prepared.generation_prompt,
            env_id=prepared.env_id,
            output_root=config.output_root,
            model=prepared.model,
            image_paths=prepared.image_paths,
        )
    except Exception as exc:
        mark_visual_review_aborted(
            scene_dir_for(config.output_root, prepared.env_id),
            prepared.visual_review_id,
            f"Generation failed before visual review: {exc}",
        )
        raise
    return finish_generate_request(config, prepared, result)[0]


def handle_behavior_run_request(
    config: StudioConfig,
    *,
    env_id: str,
    model: str = "",
    trial_ids: set[str] | None = None,
    emit: Any | None = None,
) -> dict[str, Any]:
    safe_env_id = slugify_env_id(env_id)
    if safe_env_id != env_id:
        raise ValueError("invalid environment id")
    scene_dir = scene_dir_for(config.output_root, safe_env_id)
    result = run_behavior_trials(
        scene_dir=scene_dir,
        model=model,
        trial_ids=trial_ids,
        emit=emit,
    )
    result["scene"] = load_live_scene(scene_dir)
    return result


def behavior_repair_prompt(scene_dir: Path) -> str:
    from .env_behavior_trials import load_behavior_trial_report

    report = load_behavior_trial_report(scene_dir)
    if not report:
        raise ValueError("there is no current behavior report to repair from")
    failures = [
        result
        for result in report.get("results") or []
        if isinstance(result, dict) and result.get("status") in {"failed", "inconclusive"}
    ]
    if not failures:
        raise ValueError("the current behavior report has no failed or inconclusive trials")
    compact = [
        {
            "trial_id": item.get("trial_id"),
            "instruction": item.get("instruction"),
            "status": item.get("status"),
            "termination_reason": item.get("termination_reason"),
            "objective": item.get("objective"),
            "repair_hints": item.get("repair_hints") or [],
        }
        for item in failures
    ]
    return (
        "Repair the current environment using the following code-scored behavior trial evidence. "
        "Treat an inconclusive expected-success rollout as evidence to inspect, not proof of impossibility. "
        "Make targeted changes, redefine current deterministic checks and behavior trials, validate, and finalize.\n\n"
        + json.dumps(compact, indent=2, ensure_ascii=False)[:24000]
    )


def behavior_regeneration_prompt(scene_dir: Path) -> str:
    from .env_behavior_trials import load_behavior_trial_plan

    plan = load_behavior_trial_plan(scene_dir)
    prior_prompt = str((plan or {}).get("prompt") or "").strip()
    return (
        "Regenerate the behavior trial definitions for the current finalized environment using the latest "
        "behavior controller contract. Do not change scene objects, positions, dimensions, physics, or visuals. "
        "Inspect the current spec and conversation intent, resume the scene with this request, define one or two "
        "prompt-derived affordance trials when an agent exists, including an intent_summary, validate, and finalize. Objectives must describe "
        "behaviors or prohibited counterexamples positively. Put only genuine attempt rules in constraints. Use "
        "stable/anchored targets for structural trials unless target motion is intentional.\n\n"
        f"Prior behavior-plan intent:\n{prior_prompt[:12000]}"
    )


def prepare_generate_request(config: StudioConfig, payload: dict[str, Any]) -> PreparedGeneration:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("environment must have a name")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    model = str(payload.get("model") or "").strip()
    env_id = next_env_id(config.output_root, name)
    difficulty = str(payload.get("difficulty") or "medium").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be easy, medium, or hard")
    family = str(payload.get("family") or "").strip().lower()
    if family and family not in {"barrier_route", "slalom", "push_lane", "elevation", "switch_gate", "mixed"}:
        raise ValueError("family is not a supported courtyard layout")
    raw_seed = payload.get("seed")
    if raw_seed is None:
        seed = _level_seed(env_id, prompt)
    elif isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or not 0 <= raw_seed <= 2**63 - 1:
        raise ValueError("seed must be an integer between 0 and 2^63-1")
    else:
        seed = raw_seed
    scene_dir = scene_dir_for(config.output_root, env_id)
    baseline_spec = courtyard_shell_spec(env_id, description=prompt, seed=seed)
    persist_artifacts(
        spec=baseline_spec,
        scene_dir=scene_dir,
        trace_records=[
            {
                "event": "create_courtyard_baseline",
                "env_id": env_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input": {"seed": seed},
                "output": {"status": "success", "object_count": len(baseline_spec.objects)},
            }
        ],
        render=False,
        display_name=name or env_id,
    )
    baseline_scene = load_scene(scene_dir)
    if baseline_scene is None:
        raise RuntimeError("could not persist the blank courtyard baseline")
    history = append_history_turn(config.output_root, env_id, "user", prompt)
    view_context = normalize_view_context(payload.get("view_context"))
    review = create_visual_review(
        scene_dir=scene_dir,
        env_id=env_id,
        kind="initial",
        latest_request=prompt,
        history=history,
        model=model,
        view_context=view_context,
        before_spec=baseline_scene.get("spec"),
        before_visual_scene=baseline_scene.get("visual_scene"),
    )
    stored_view_context = persist_studio_view_context(
        scene_dir=scene_dir,
        review_id=str(review["review_id"]),
        prompt=prompt,
        view_context=view_context,
        image_data_url=str(payload.get("submitted_view_image") or ""),
    )
    image_path = submitted_view_image_path(scene_dir, stored_view_context)
    generation_prompt = build_generation_prompt(
        env_id=env_id,
        user_prompt=prompt,
        difficulty=difficulty,
        seed=seed,
        family=family,
        view_context=view_context,
        has_view_image=image_path is not None,
    )
    return PreparedGeneration(
        env_id=env_id,
        prompt=prompt,
        model=model,
        generation_prompt=generation_prompt,
        visual_review_id=str(review["review_id"]),
        image_paths=(image_path,) if image_path is not None else (),
    )


def finish_generate_request(
    config: StudioConfig,
    prepared: PreparedGeneration,
    result: dict[str, Any],
    *,
    activity_events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    assistant_text = assistant_text_from_events(
        result.get("events") or [],
        fallback="Created and finalized the environment.",
    )
    append_history_turn(
        config.output_root,
        prepared.env_id,
        "assistant",
        assistant_text,
        activity=activity_events or public_activity_from_events(result.get("events") or []),
    )
    scene = result.get("scene") or load_live_scene(config.output_root / prepared.env_id)
    scene = _finish_visual_review_record(
        config,
        env_id=prepared.env_id,
        review_id=prepared.visual_review_id,
        scene=scene,
    )
    scene = with_scene_history(scene, config.output_root, prepared.env_id)
    response = {
        "status": "success" if scene and scene.get("status") == "finalized" else "incomplete",
        "env_id": prepared.env_id,
        "visual_review_id": prepared.visual_review_id,
        "scene": scene,
        "history": read_history(config.output_root, prepared.env_id),
        "events": result.get("events") or [],
        "stderr": result.get("stderr") or [],
    }
    return response, assistant_text


def handle_generate_stream_request(
    config: StudioConfig,
    payload: dict[str, Any],
    emit: Any,
) -> None:
    prepared = prepare_generate_request(config, payload)
    activity_events: list[dict[str, Any]] = []
    emitted_text = False
    last_scene_key = ""

    def emit_progress(progress: dict[str, Any]) -> None:
        normalized = normalize_activity_events([progress])
        if not normalized:
            return
        event = normalized[0]
        if activity_events and activity_events[-1] == event:
            return
        activity_events.append(event)
        emit("progress", event)

    def emit_scene_if_changed() -> None:
        nonlocal last_scene_key
        scene = load_live_scene(config.output_root / prepared.env_id)
        if not scene:
            return
        scene_key = json.dumps(scene.get("spec") or {}, sort_keys=True, separators=(",", ":"))
        if scene_key == last_scene_key:
            return
        last_scene_key = scene_key
        scene = with_scene_history(scene, config.output_root, prepared.env_id)
        emit("scene", {"phase": scene.get("status") or "draft", "scene": scene})

    emit(
        "system",
        {
            "env_id": prepared.env_id,
            "message": "Starting streamed generation.",
        },
    )
    emit_progress(
        {
            "type": "progress",
            "label": "Preparing environment",
            "message": "Reserved the environment and started Codex.",
        }
    )
    emit_scene_if_changed()

    def on_event(event: dict[str, Any]) -> None:
        nonlocal emitted_text
        event_type = str(event.get("type") or "").strip().lower()
        message = str(event.get("message") or "").strip()
        if event_type in {"agent_message", "assistant", "assistant_message", "message", "response_item"} and message:
            emitted_text = True
            emit("text", {"delta": f"{message}\n"})
        else:
            progress = progress_update_for_event(event)
            if progress:
                emit_progress(progress)
        emit_scene_if_changed()

    try:
        result = run_codex_generation_stream(
            prompt=prepared.generation_prompt,
            env_id=prepared.env_id,
            output_root=config.output_root,
            model=prepared.model,
            on_event=on_event,
            image_paths=prepared.image_paths,
        )
    except Exception as exc:
        mark_visual_review_aborted(
            scene_dir_for(config.output_root, prepared.env_id),
            prepared.visual_review_id,
            f"Generation failed before visual review: {exc}",
        )
        raise
    emit_scene_if_changed()
    emit_progress({"type": "progress", "label": "Refreshing scene", "message": "Loading finalized artifacts and history."})
    response, assistant_text = finish_generate_request(
        config,
        prepared,
        result,
        activity_events=activity_events,
    )
    if not emitted_text:
        emit("text", {"delta": assistant_text})
    emit("done", response)


def handle_revise_request(
    config: StudioConfig,
    raw_env_id: str,
    payload: dict[str, Any],
    *,
    revision_evidence: str = "",
    evidence_image_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    prepared = prepare_revise_request(
        config,
        raw_env_id,
        payload,
        revision_evidence=revision_evidence,
        evidence_image_paths=evidence_image_paths,
    )
    try:
        result = run_codex_generation(
            prompt=prepared.revision_prompt,
            env_id=prepared.env_id,
            output_root=config.output_root,
            model=prepared.model,
            image_paths=prepared.image_paths,
        )
    except Exception as exc:
        mark_visual_review_aborted(
            scene_dir_for(config.output_root, prepared.env_id),
            prepared.visual_review_id,
            f"Revision failed before visual review: {exc}",
        )
        raise
    response, _assistant_text = finish_revise_request(config, prepared, result)
    return response


def handle_revise_stream_request(
    config: StudioConfig,
    raw_env_id: str,
    payload: dict[str, Any],
    emit: Any,
    *,
    revision_evidence: str = "",
    evidence_image_paths: tuple[Path, ...] = (),
) -> None:
    prepared = prepare_revise_request(
        config,
        raw_env_id,
        payload,
        revision_evidence=revision_evidence,
        evidence_image_paths=evidence_image_paths,
    )
    activity_events: list[dict[str, Any]] = []
    emitted_text = False

    def emit_progress(progress: dict[str, Any]) -> None:
        normalized = normalize_activity_events([progress])
        if not normalized:
            return
        event = normalized[0]
        if activity_events and activity_events[-1] == event:
            return
        activity_events.append(event)
        emit("progress", event)

    emit(
        "system",
        {
            "env_id": prepared.env_id,
            "outputRoot": str(config.output_root),
            "message": "Starting streamed revision.",
        },
    )
    emit_progress({"type": "progress", "label": "Preparing revision", "message": "Loaded scene context and conversation history."})

    def on_event(event: dict[str, Any]) -> None:
        nonlocal emitted_text
        event_type = str(event.get("type") or "").strip().lower()
        message = str(event.get("message") or "").strip()
        if event_type in {"agent_message", "assistant", "assistant_message", "message", "response_item"} and message:
            emitted_text = True
            emit("text", {"delta": f"{message}\n"})
            return
        progress = progress_update_for_event(event)
        if progress:
            emit_progress(progress)

    try:
        result = run_codex_generation_stream(
            prompt=prepared.revision_prompt,
            env_id=prepared.env_id,
            output_root=config.output_root,
            model=prepared.model,
            on_event=on_event,
            image_paths=prepared.image_paths,
        )
    except Exception as exc:
        mark_visual_review_aborted(
            scene_dir_for(config.output_root, prepared.env_id),
            prepared.visual_review_id,
            f"Revision failed before visual review: {exc}",
        )
        raise
    emit_progress({"type": "progress", "label": "Refreshing scene", "message": "Loading updated artifacts and history."})
    response, assistant_text = finish_revise_request(config, prepared, result, activity_events=activity_events)
    if not emitted_text:
        emit("text", {"delta": assistant_text})
    emit("done", response)


def prepare_revise_request(
    config: StudioConfig,
    raw_env_id: str,
    payload: dict[str, Any],
    *,
    revision_evidence: str = "",
    evidence_image_paths: tuple[Path, ...] = (),
) -> PreparedRevision:
    env_id = slugify_env_id(raw_env_id)
    if env_id != raw_env_id:
        raise ValueError("invalid environment id")
    prompt = str(payload.get("message") or payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("message is required")
    model = str(payload.get("model") or "").strip()
    scene_dir = scene_dir_for(config.output_root, env_id)
    scene = load_scene(scene_dir)
    if scene is None:
        raise ValueError(f"scene {env_id!r} not found")
    context_images = _validated_revision_evidence_paths(scene_dir, evidence_image_paths)
    request_history = normalize_history_turns(payload.get("history"))
    if request_history:
        write_history(config.output_root, env_id, request_history)
    history = read_history(config.output_root, env_id)
    view_context = normalize_view_context(payload.get("view_context"))
    review = create_visual_review(
        scene_dir=scene_dir,
        env_id=env_id,
        kind="revision",
        latest_request=prompt,
        history=history,
        model=model,
        view_context=view_context,
        before_spec=scene.get("spec"),
        before_visual_scene=scene.get("visual_scene"),
    )
    stored_view_context = persist_studio_view_context(
        scene_dir=scene_dir,
        review_id=str(review["review_id"]),
        prompt=prompt,
        view_context=view_context,
        image_data_url=str(payload.get("submitted_view_image") or ""),
    )
    image_path = submitted_view_image_path(scene_dir, stored_view_context)
    append_history_turn(config.output_root, env_id, "user", prompt)
    revision_prompt = build_revision_prompt(
        env_id=env_id,
        user_prompt=prompt,
        scene=scene,
        history=history,
        view_context=view_context,
        has_view_image=image_path is not None,
        revision_evidence=revision_evidence,
    )
    return PreparedRevision(
        env_id=env_id,
        prompt=prompt,
        model=model,
        revision_prompt=revision_prompt,
        visual_review_id=str(review["review_id"]),
        image_paths=context_images + ((image_path,) if image_path is not None else ()),
    )


def _validated_revision_evidence_paths(scene_dir: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if len(paths) > 12:
        raise ValueError("revision evidence exceeds the 12-image limit")
    root = scene_dir.resolve()
    validated: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("revision evidence must belong to the current environment") from exc
        if not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError("revision evidence must be an existing PNG")
        if path not in validated:
            validated.append(path)
    return tuple(validated)


def finish_revise_request(
    config: StudioConfig,
    prepared: PreparedRevision,
    result: dict[str, Any],
    *,
    activity_events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    assistant_text = assistant_text_from_events(
        result.get("events") or [],
        fallback="Applied the requested changes and finalized the environment.",
    )
    append_history_turn(
        config.output_root,
        prepared.env_id,
        "assistant",
        assistant_text,
        activity=activity_events or public_activity_from_events(result.get("events") or []),
    )
    updated_scene = load_scene(config.output_root / prepared.env_id) or result.get("scene")
    updated_scene = _finish_visual_review_record(
        config,
        env_id=prepared.env_id,
        review_id=prepared.visual_review_id,
        scene=updated_scene,
    )
    updated_scene = with_scene_history(updated_scene, config.output_root, prepared.env_id)
    response = {
        "status": "success" if updated_scene else "incomplete",
        "env_id": prepared.env_id,
        "visual_review_id": prepared.visual_review_id,
        "scene": updated_scene,
        "history": read_history(config.output_root, prepared.env_id),
        "events": result.get("events") or [],
        "stderr": result.get("stderr") or [],
    }
    return response, assistant_text


def _finish_visual_review_record(
    config: StudioConfig,
    *,
    env_id: str,
    review_id: str,
    scene: dict[str, Any] | None,
) -> dict[str, Any] | None:
    spec = scene.get("spec") if isinstance(scene, dict) else None
    visual_scene = scene.get("visual_scene") if isinstance(scene, dict) else None
    if (
        isinstance(scene, dict)
        and scene.get("status") == "finalized"
        and isinstance(spec, dict)
        and isinstance(visual_scene, dict)
    ):
        mark_visual_review_ready(
            scene_dir_for(config.output_root, env_id),
            review_id,
            after_spec=spec,
            after_visual_scene=visual_scene,
        )
        return load_live_scene(config.output_root / env_id) or scene
    mark_visual_review_aborted(
        scene_dir_for(config.output_root, env_id),
        review_id,
        "Environment was not finalized, so visual review was skipped.",
    )
    return scene


def parse_args(argv: list[str] | None = None) -> StudioConfig:
    parser = argparse.ArgumentParser(description="Run the Environment Generation server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3033)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    return StudioConfig(
        host=args.host,
        port=args.port,
        output_root=args.output_root.expanduser().resolve(),
        open_browser=not args.no_open,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    config.output_root.mkdir(parents=True, exist_ok=True)
    for scene_dir in config.output_root.iterdir():
        if scene_dir.is_dir() and not scene_dir.name.startswith("."):
            recover_abandoned_behavior_runs(scene_dir)
            recover_abandoned_task_runs(scene_dir)
            recover_interrupted_task_compilations(scene_dir)
            TASK_ORACLE_SESSIONS.recover_scene(scene_dir)
    server = ThreadingHTTPServer((config.host, config.port), StudioHandler)
    server.studio_config = config  # type: ignore[attr-defined]
    url = f"http://{config.host}:{config.port}"
    print(f"Environment Generation running at {url}")
    if config.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
