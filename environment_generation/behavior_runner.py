"""Studio orchestration for isolated child-agent behavior trials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import ENV_SPEC_FILENAME, VISUAL_SCENE_FILENAME, WORLD_XML_FILENAME
from .behavior_evidence import select_behavior_evidence, significant_frame_events
from .behavior_preflight import prepare_behavior_trial
from .behavior_trial import FRAME_MANIFEST_FILENAME, read_action_log, replay_behavior_actions
from .env_behavior_trials import (
    BEHAVIOR_CONTROLLER_VERSION,
    ENV_BEHAVIOR_TRIALS_DIRNAME,
    behavior_plan_hash,
    build_behavior_trial_report,
    clear_behavior_trial_report,
    load_behavior_trial_plan,
    load_behavior_trial_report,
    write_behavior_trial_plan,
    write_behavior_trial_report,
)
from .env_verification import spec_hash
from .runtime_config import rendering_subprocess_env, runtime_env_key, runtime_env_value


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_NAME = "environment-generation"
DEFAULT_CHILD_TIMEOUT_SECONDS = 10 * 60
MIN_CHILD_TIMEOUT_SECONDS = 60
MAX_CHILD_TIMEOUT_SECONDS = 30 * 60
MAX_CHILD_OUTPUT_CHARS = 12000
MAX_EVIDENCE_FRAMES = 6
ACTIVE_RUN_STATUSES = frozenset({"preparing", "running", "replaying", "evaluating"})
_RUN_ADMISSION_LOCK = threading.Lock()
_REPORT_WRITE_LOCK = threading.Lock()
BehaviorEmit = Callable[[str, dict[str, Any]], None]
ChildExecutor = Callable[[Path, dict[str, Any], str, Path, Path, BehaviorEmit | None], dict[str, Any]]


def _child_timeout_seconds() -> int:
    raw_value = str(runtime_env_value("BEHAVIOR_TIMEOUT_SECONDS") or "").strip()
    if not raw_value:
        return DEFAULT_CHILD_TIMEOUT_SECONDS
    try:
        configured = int(raw_value)
    except ValueError:
        return DEFAULT_CHILD_TIMEOUT_SECONDS
    return min(MAX_CHILD_TIMEOUT_SECONDS, max(MIN_CHILD_TIMEOUT_SECONDS, configured))


def run_behavior_trials(
    *,
    scene_dir: Path,
    model: str = "",
    trial_ids: set[str] | None = None,
    emit: BehaviorEmit | None = None,
    child_executor: ChildExecutor | None = None,
) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    model = str(runtime_env_value("BEHAVIOR_MODEL") or model or "").strip()
    spec = _read_json(scene_dir / ENV_SPEC_FILENAME)
    plan = load_behavior_trial_plan(scene_dir)
    if not spec:
        raise ValueError("finalized env_spec_3d.json is required before running behavior trials")
    if not any(obj.get("semantic_type") == "agent" for obj in spec.get("objects") or [] if isinstance(obj, dict)):
        raise ValueError("behavior trials are not applicable because the scene has no authored agent")
    if not plan:
        raise ValueError("behavior trial plan is missing")
    if plan.get("controller_version") != BEHAVIOR_CONTROLLER_VERSION:
        raise ValueError("behavior trial plan uses an older controller contract; revise or regenerate the plan")
    current_spec_hash = spec_hash(spec)
    if plan.get("draft_hash") != current_spec_hash:
        raise ValueError("behavior trial plan is stale for the current scene")
    all_trials = [trial for trial in plan.get("trials") or [] if isinstance(trial, dict)]
    selected_ids = {str(value) for value in (trial_ids or set()) if value}
    trials = [trial for trial in all_trials if not selected_ids or str(trial.get("id")) in selected_ids]
    if not trials:
        raise ValueError("no matching behavior trials were selected")
    requested_trial_ids = {str(trial.get("id")) for trial in trials}
    with _RUN_ADMISSION_LOCK:
        recover_abandoned_behavior_runs(scene_dir)
        active_trial_ids = {
            str(trial_id)
            for active in _active_manifests(scene_dir)
            for trial_id in active.get("trial_ids") or []
        }
        overlapping = sorted(requested_trial_ids & active_trial_ids)
        if overlapping:
            names = ", ".join(overlapping)
            raise ValueError(f"behavior test already running: {names}")

        run_id = _new_run_id(scene_dir)
        run_dir = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME / run_id
        snapshot_dir = run_dir / "snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for filename in (ENV_SPEC_FILENAME, WORLD_XML_FILENAME, VISUAL_SCENE_FILENAME):
            source = scene_dir / filename
            if source.is_file():
                shutil.copy2(source, snapshot_dir / filename)
        _write_json(snapshot_dir / "plan.json", plan)
        manifest = {
            "schema_version": "1.0",
            "controller_version": BEHAVIOR_CONTROLLER_VERSION,
            "run_id": run_id,
            "env_id": str(spec.get("id") or scene_dir.name),
            "created_at": _now(),
            "pid": os.getpid(),
            "status": "preparing",
            "env_spec_hash": current_spec_hash,
            "behavior_plan_hash": behavior_plan_hash(plan),
            "trial_ids": [str(trial.get("id")) for trial in trials],
            "model": model or "default",
        }
        _write_json(run_dir / "manifest.json", manifest)
    _emit(emit, "behavior_trials", {"status": "preparing", "run_id": run_id, "manifest": manifest})

    executor = child_executor or _run_codex_child
    results: list[dict[str, Any]] = []
    try:
        manifest["status"] = "running"
        _write_json(run_dir / "manifest.json", manifest)
        for trial in trials:
            trial_id = str(trial.get("id") or "trial")
            trial_dir = run_dir / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            contextual_trial = {
                **trial,
                "environment_request": plan.get("prompt") or "",
                "scene_description": spec.get("description") or "",
                "fallback": bool(plan.get("fallback")),
            }
            runtime_trial, preflight = prepare_behavior_trial(
                scene_dir=snapshot_dir,
                trial=contextual_trial,
            )
            _write_json(trial_dir / "preflight.json", preflight)
            _write_json(trial_dir / "runtime_trial.json", runtime_trial)
            _emit(
                emit,
                "behavior_trials",
                {
                    "status": "preflight",
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "preflight": preflight,
                },
            )
            if preflight["status"] != "ready":
                result = _invalid_setup_result(runtime_trial, preflight)
                _write_json(trial_dir / "trajectory.json", {"trial_id": trial_id, "frames": []})
                _write_json(trial_dir / "result.json", result)
                results.append(result)
                _emit(
                    emit,
                    "behavior_trials",
                    {"status": "invalid_setup", "run_id": run_id, "trial_id": trial_id, "result": result},
                )
                continue
            action_log = trial_dir / "child_actions.jsonl"
            observation_log = trial_dir / "child_observations.jsonl"
            child_frame_dir = trial_dir / "child_frames"
            replay_frame_dir = trial_dir / "replay_frames"
            _emit(
                emit,
                "behavior_trials",
                {
                    "status": "child_running",
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "instruction": runtime_trial.get("instruction"),
                },
            )
            child = executor(snapshot_dir, runtime_trial, model, action_log, child_frame_dir, emit)
            actions = read_action_log(action_log)
            _write_json(trial_dir / "actions.json", actions)
            manifest["status"] = "replaying"
            _write_json(run_dir / "manifest.json", manifest)
            _emit(
                emit,
                "behavior_trials",
                {"status": "replaying", "run_id": run_id, "trial_id": trial_id, "action_count": len(actions)},
            )
            if not actions and int(child.get("exit_code") or 0) != 0:
                result = _child_error_result(runtime_trial, child)
            else:
                result = replay_behavior_actions(
                    scene_dir=snapshot_dir,
                    trial=runtime_trial,
                    actions=actions,
                    frame_dir=replay_frame_dir,
                )
                result["child_summary"] = str(child.get("summary") or "")[:MAX_CHILD_OUTPUT_CHARS]
                result["child_exit_code"] = int(child.get("exit_code") or 0)
                result["child_stderr"] = [str(line)[:500] for line in child.get("stderr") or []][-20:]
            observation_count = _jsonl_record_count(observation_log)
            result["child_observation_count"] = observation_count
            if observation_count:
                result["child_observations_url"] = (
                    f"/generated/{scene_dir.name}/{ENV_BEHAVIOR_TRIALS_DIRNAME}/"
                    f"{run_id}/{trial_id}/{observation_log.name}"
                )
            result["preflight"] = preflight
            trajectory = result.pop("trajectory", [])
            _write_json(trial_dir / "trajectory.json", {"trial_id": trial_id, "frames": trajectory})
            result["trajectory_url"] = (
                f"/generated/{scene_dir.name}/{ENV_BEHAVIOR_TRIALS_DIRNAME}/{run_id}/{trial_id}/trajectory.json"
            )
            replay_verified = _frame_capture_complete(
                replay_frame_dir,
                expected_step=int(result.get("steps_used") or 0),
            )
            evidence_source_dir = replay_frame_dir if replay_verified else child_frame_dir
            evidence_source = "authoritative_replay" if replay_verified else "child_policy"
            result["evidence_schema_version"] = 2
            result["evidence_source"] = evidence_source
            result["evidence_replay_verified"] = replay_verified
            result["evidence_frames"] = _persist_evidence_frames(
                scene_dir=scene_dir,
                source_dir=evidence_source_dir,
                target_dir=trial_dir / "evidence",
                run_id=run_id,
                trial_id=trial_id,
                source=evidence_source,
                replay_verified=replay_verified,
            )
            _write_json(trial_dir / "result.json", result)
            results.append(result)
            _emit(
                emit,
                "behavior_trials",
                {"status": "child_done", "run_id": run_id, "trial_id": trial_id, "result": result},
            )

        manifest["status"] = "evaluating"
        _write_json(run_dir / "manifest.json", manifest)
        with _REPORT_WRITE_LOCK:
            current_spec = _read_json(scene_dir / ENV_SPEC_FILENAME)
            current_plan = load_behavior_trial_plan(scene_dir)
            stale = (
                not current_spec
                or not current_plan
                or spec_hash(current_spec) != current_spec_hash
                or behavior_plan_hash(current_plan) != behavior_plan_hash(plan)
            )
            report_results = _merge_behavior_results(
                all_trials=all_trials,
                new_results=results,
                current_report=load_behavior_trial_report(scene_dir) if not stale else None,
                env_spec_hash=current_spec_hash,
                plan_hash=behavior_plan_hash(plan),
                merge_current=bool(selected_ids),
            )
            report = build_behavior_trial_report(
                env_id=str(spec.get("id") or scene_dir.name),
                plan=plan,
                env_spec_hash=current_spec_hash,
                run_id=run_id,
                results=report_results,
                model=model,
            )
            report["stale"] = stale
            _write_json(run_dir / "report.json", report)
            if not stale:
                write_behavior_trial_report(scene_dir, report)
            manifest.update(
                {
                    "status": report["status"],
                    "completed_at": _now(),
                    "stale": stale,
                    "pid": None,
                }
            )
            _write_json(run_dir / "manifest.json", manifest)
        _emit(emit, "behavior_trials", {"status": report["status"], "run_id": run_id, "report": report})
        return {"status": "success", "run_id": run_id, "report": report, "manifest": manifest}
    except Exception as exc:
        manifest.update({"status": "error", "completed_at": _now(), "error": str(exc), "pid": None})
        _write_json(run_dir / "manifest.json", manifest)
        _emit(emit, "behavior_trials", {"status": "error", "run_id": run_id, "error": str(exc)})
        raise


def dismiss_behavior_trials(*, scene_dir: Path, trial_ids: set[str] | None = None) -> dict[str, Any]:
    plan = load_behavior_trial_plan(scene_dir)
    if not plan:
        raise ValueError("behavior trial plan is missing")
    selected = {str(value) for value in (trial_ids or set()) if value}
    known = {str(trial.get("id")) for trial in plan.get("trials") or [] if isinstance(trial, dict)}
    if selected and not selected.issubset(known):
        raise ValueError("one or more behavior trial ids were not found")
    dismissed = known if not selected else selected
    plan["dismissed_trial_ids"] = sorted(dismissed)
    plan["dismissed"] = dismissed == known
    plan["dismissed_at"] = _now()
    write_behavior_trial_plan(scene_dir, plan)
    clear_behavior_trial_report(scene_dir)
    return {"status": "dismissed", "dismissed_trial_ids": sorted(dismissed), "plan": plan}


def _run_codex_child(
    scene_dir: Path,
    trial: dict[str, Any],
    model: str,
    action_log: Path,
    frame_dir: Path,
    emit: BehaviorEmit | None,
) -> dict[str, Any]:
    del emit
    if shutil.which("codex") is None:
        raise RuntimeError("'codex' is not on PATH; install/login to the Codex CLI first")
    with tempfile.TemporaryDirectory(prefix="environment-generation-behavior-child-") as temporary:
        temp_dir = Path(temporary)
        output_path = temp_dir / "last_message.txt"
        command = sys.executable
        mcp_args = ["-m", "environment_generation.mcp_server"]
        mcp_env = {
            **rendering_subprocess_env(),
            "PYTHONPATH": str(PROJECT_ROOT),
            runtime_env_key("BEHAVIOR_SCENE_DIR"): str(scene_dir),
            runtime_env_key("BEHAVIOR_TRIAL_JSON"): json.dumps(trial, separators=(",", ":")),
            runtime_env_key("BEHAVIOR_ACTION_LOG"): str(action_log),
            runtime_env_key("BEHAVIOR_OBSERVATION_LOG"): str(
                action_log.with_name("child_observations.jsonl")
            ),
            runtime_env_key("BEHAVIOR_FRAME_DIR"): str(frame_dir),
        }
        args = [
            "codex",
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
            "-c",
            f"mcp_servers.{MCP_SERVER_NAME}.command={_toml_value(command)}",
            "-c",
            f"mcp_servers.{MCP_SERVER_NAME}.args={_toml_value(mcp_args)}",
            "-c",
            f'mcp_servers.{MCP_SERVER_NAME}.default_tools_approval_mode="approve"',
        ]
        for key, value in mcp_env.items():
            args.extend(["-c", f"mcp_servers.{MCP_SERVER_NAME}.env.{key}={_toml_value(value)}"])
        if model:
            args.extend(["-m", model])
        args.extend(["--output-last-message", str(output_path), _child_prompt(trial)])
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        timeout_seconds = _child_timeout_seconds()
        try:
            completed = subprocess.run(
                args,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stderr = completed.stderr.splitlines()[-20:]
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stderr = [f"Child behavior trial exceeded {timeout_seconds} seconds."]
            if exc.stderr:
                stderr.extend(str(exc.stderr).splitlines()[-10:])
        try:
            summary = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            summary = ""
        return {"exit_code": exit_code, "summary": summary[:MAX_CHILD_OUTPUT_CHARS], "stderr": stderr}


def _child_prompt(trial: dict[str, Any]) -> str:
    evidence = json.dumps(
        {
            "trial_id": trial.get("id"),
            "instruction": trial.get("instruction"),
            "expected_outcome": trial.get("expected_outcome"),
            "objective": trial.get("objective"),
            "constraints": trial.get("constraints"),
            "navigation": trial.get("navigation"),
            "environment_request": trial.get("environment_request"),
            "scene_description": trial.get("scene_description"),
            "fallback": bool(trial.get("fallback")),
            "max_steps": trial.get("max_steps"),
            "max_total_steps": trial.get("max_total_steps"),
            "max_resets": trial.get("max_resets"),
        },
        indent=2,
    )
    return f"""\
You are a child Codex agent controlling one immutable Environment Generation MuJoCo trial.
The typed objective below is the only definition of success. Do not assume the
scene is a goal-reaching level: a trial may require locomotion, positioning,
climbing, pushing, contact, mechanism activation, stability, or an active search
for a prohibited counterexample.

Trial:
{evidence}

Use only start_behavior_trial, observe_behavior_trial, act_behavior_trial,
reset_behavior_trial, and stop_behavior_trial. Do not edit files or environments.
Start and observe before acting. Use the first-person frame and the structured
telemetry together:
- objective_focus identifies the current unmet typed assertion and distinguishes
  controllable actor, manipulated subject, destination target, and navigation target.
  Re-read it after each action because ordered checks advance.
- nearby_objects gives world pose, dimensions, bounds, camera-relative direction,
  clearance, contacts, and semantic affordances for objective targets, risks, and
  surrounding geometry.
- route_guidance is optional, advisory ground-approach help. When available, steer
  toward next_waypoint_relative and re-observe at turns. It is not a proof of task
  feasibility and it does not solve jumping, pushing, or interaction for you.
- interaction_guidance is present for subject-to-target manipulation. Approach its
  predicate-specific, shape-aware staging point, align through the subject toward
  desired_subject_center, then push manually in short batches. Its geometric fields
  are advisory: only objective checks establish completion. If the phase is
  awaiting_objective_evidence, re-observe or adjust instead of assuming success.
- navigation is the direct bearing to the current target. It may cross blockers or
  hazards, so do not blindly follow it when route_guidance says direct_path_blocked.
- action_guidance recommends a maximum frame batch near failure zones, route turns,
  and interaction boundaries. Respect it; otherwise normally use 8-24 frames.

For a long ground approach, prefer one bounded
act_behavior_trial(assist="ground_route", frames=240) call over dozens of manual
micro-actions. This reusable controller follows the current sparse route, recovers
clearance, and records every low-level action for exact parent replay. It stops at
solid-object interaction range, so inspect and switch to manual actions for pushing,
jumping, climbing, contact strategy, or route search. It is unavailable for
object-free objectives and does not determine whether a trial passes.
For a multi-stage route, you may pass target_id for any known scene object. For
example, if a closed gate blocks the objective, use the mechanism's trigger_id as
the assisted target, let the switch latch and gate open, then resume assistance
toward the objective. This changes navigation only; the typed objective remains the
sole success criterion.

For camera-relative telemetry, positive right means steer right. A positive
heading_error_degrees means turn the camera right with positive look_x; a negative
value means turn left. Use fine 2-6 frame control near hazards, target boundaries,
objects being pushed, jump takeoffs, and landing surfaces. Confirm progress from
typed objective metrics rather than narration. Settle before jumping and do not
repeatedly jump while airborne. Goal, hazard, switch, and target-region sensors
only report entry and exit; they do not end or reset an attempt by themselves. The
typed objective and constraints decide whether a sensor event is required,
forbidden, or irrelevant. Leaving the playable bounds remains a hard safety
failure. If terminal is true and trial_satisfied is false, use
reset_behavior_trial when a reset remains. Each reset receives a fresh per-attempt step budget,
so use distinct strategies when a reset is available. For should_not_succeed,
actively search for the prohibited counterexample while honoring constraints;
merely waiting is not useful evidence. Stop when the objective is demonstrated or
all attempts are genuinely exhausted. Finish with a short factual summary. Never
claim impossibility from a bounded failed search.
"""


def _persist_evidence_frames(
    *,
    scene_dir: Path,
    source_dir: Path,
    target_dir: Path,
    run_id: str,
    trial_id: str,
    source: str,
    replay_verified: bool,
) -> list[dict[str, Any]]:
    if not source_dir.is_dir():
        return []
    paths = sorted(source_dir.glob("*.png"))
    manifest = _read_frame_manifest(source_dir / FRAME_MANIFEST_FILENAME)
    records_by_name = {
        Path(str(record.get("path") or "")).name: record
        for record in manifest
        if record.get("path")
    }
    selections = select_behavior_evidence(
        paths,
        records_by_name,
        limit=MAX_EVIDENCE_FRAMES,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    values = []
    for index, selection in enumerate(selections):
        source_path = selection.path
        target = target_dir / f"frame_{index:03d}.png"
        shutil.copy2(source_path, target)
        metadata = selection.record
        values.append(
            {
                "index": index,
                "path": str(target),
                "url": (
                    f"/generated/{scene_dir.name}/{ENV_BEHAVIOR_TRIALS_DIRNAME}/"
                    f"{run_id}/{trial_id}/evidence/{target.name}"
                ),
                "step": metadata.get("total_step"),
                "attempt": metadata.get("attempt"),
                "kind": selection.kind,
                "label": selection.label,
                "focus_ids": list(selection.focus_ids),
                "source": source,
                "replay_verified": replay_verified,
                "events": significant_frame_events(metadata),
                "camera": metadata.get("camera") or {},
                "agent": metadata.get("agent") or {},
                "navigation": metadata.get("navigation") or {},
                "objective_focus": metadata.get("objective_focus") or {},
                "interaction_guidance": metadata.get("interaction_guidance") or {},
                "objective": metadata.get("attempt_objective") or metadata.get("objective") or {},
            }
        )
    return values


def _read_frame_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    values = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _frame_capture_complete(frame_dir: Path, *, expected_step: int) -> bool:
    if not frame_dir.is_dir():
        return False
    records = _read_frame_manifest(frame_dir / FRAME_MANIFEST_FILENAME)
    if not records or not any(frame_dir.glob("*.png")):
        return False
    try:
        final_step = int(records[-1].get("total_step") or 0)
    except (TypeError, ValueError):
        return False
    return final_step == expected_step


def _child_error_result(trial: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial.get("id"),
        "instruction": trial.get("instruction"),
        "expected_outcome": trial.get("expected_outcome"),
        "severity": trial.get("severity") or "critical",
        "status": "error",
        "passed": False,
        "objective": {"satisfied": False, "checks": [], "passed_count": 0, "total_count": 0},
        "reward": 0.0,
        "termination_reason": "child_error",
        "steps_used": 0,
        "reset_count": 0,
        "attempt_count": 0,
        "actions": [],
        "events": [],
        "attempts": [],
        "final_state": {},
        "child_summary": str(child.get("summary") or "")[:MAX_CHILD_OUTPUT_CHARS],
        "child_exit_code": int(child.get("exit_code") or 1),
        "child_stderr": [str(line)[:500] for line in child.get("stderr") or []][-20:],
        "repair_hints": ["Retry the child trial after resolving the execution error."],
    }


def _invalid_setup_result(trial: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial.get("id"),
        "instruction": trial.get("instruction"),
        "expected_outcome": trial.get("expected_outcome"),
        "severity": trial.get("severity") or "critical",
        "status": "invalid_setup",
        "passed": False,
        "non_failure": False,
        "objective": {"satisfied": False, "checks": [], "passed_count": 0, "total_count": 0},
        "constraints": {"satisfied": False, "checks": [], "passed_count": 0, "total_count": 0},
        "reward": 0.0,
        "termination_reason": "preflight_failed",
        "steps_used": 0,
        "reset_count": 0,
        "attempt_count": 0,
        "actions": [],
        "events": [],
        "attempts": [],
        "final_state": {},
        "child_summary": "",
        "child_exit_code": 0,
        "child_stderr": [],
        "repair_hints": preflight.get("repair_hints") or [],
        "preflight": preflight,
        "evidence_frames": [],
    }


def _merge_behavior_results(
    *,
    all_trials: list[dict[str, Any]],
    new_results: list[dict[str, Any]],
    current_report: dict[str, Any] | None,
    env_spec_hash: str,
    plan_hash: str,
    merge_current: bool,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if (
        merge_current
        and isinstance(current_report, dict)
        and current_report.get("env_spec_hash") == env_spec_hash
        and current_report.get("behavior_plan_hash") == plan_hash
    ):
        merged.update(
            {
                str(item.get("trial_id")): item
                for item in current_report.get("results") or []
                if isinstance(item, dict) and item.get("trial_id")
            }
        )
    merged.update(
        {
            str(item.get("trial_id")): item
            for item in new_results
            if isinstance(item, dict) and item.get("trial_id")
        }
    )
    return [
        merged[trial_id]
        for trial in all_trials
        if (trial_id := str(trial.get("id") or "")) in merged
    ]


def _active_manifests(scene_dir: Path) -> list[dict[str, Any]]:
    root = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME
    if not root.is_dir():
        return []
    active: list[dict[str, Any]] = []
    for path in root.glob("*/manifest.json"):
        manifest = _read_json(path)
        if (
            manifest
            and manifest.get("status") in ACTIVE_RUN_STATUSES
            and _process_is_alive(manifest.get("pid"))
        ):
            active.append(manifest)
    return sorted(active, key=lambda item: str(item.get("created_at") or ""))


def _active_manifest(scene_dir: Path) -> dict[str, Any] | None:
    active = _active_manifests(scene_dir)
    return active[-1] if active else None


def recover_abandoned_behavior_runs(scene_dir: Path) -> None:
    root = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME
    if not root.is_dir():
        return
    for path in root.glob("*/manifest.json"):
        manifest = _read_json(path)
        if not manifest or manifest.get("status") not in ACTIVE_RUN_STATUSES:
            continue
        if _process_is_alive(manifest.get("pid")):
            continue
        manifest.update(
            {
                "status": "error",
                "completed_at": _now(),
                "error": "Studio restarted before this behavior run completed.",
                "pid": None,
            }
        )
        _write_json(path, manifest)


def _process_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _new_run_id(scene_dir: Path) -> str:
    root = scene_dir / ENV_BEHAVIOR_TRIALS_DIRNAME
    index = len([path for path in root.iterdir() if path.is_dir()]) + 1 if root.is_dir() else 1
    return f"run-{index:04d}-{uuid.uuid4().hex[:8]}"


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _jsonl_record_count(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            count += 1
    return count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _emit(emit: BehaviorEmit | None, event: str, data: dict[str, Any]) -> None:
    if emit is not None:
        try:
            emit(event, data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
