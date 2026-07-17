from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import PREVIEW_FILENAMES, append_trace_record, artifact_paths, persist_artifacts, persist_draft_spec
from .builder import BuilderValidationError, EnvSpec3DBuilder
from .env_behavior_trials import (
    behavior_plan_decision_issues,
    behavior_prompt_hash,
    behavior_trial_summary,
    clear_behavior_trial_report,
    fallback_locomotion_plan,
    load_behavior_trial_plan,
    load_behavior_trial_report,
    normalize_behavior_trial_plan,
    write_behavior_trial_plan,
)
from .env_verification import (
    EnvVerificationError,
    clear_env_verification_report,
    env_verification_summary,
    finalization_block,
    load_env_verification_plan,
    load_env_verification_report,
    normalize_env_verification_plan,
    run_env_verification,
    spec_hash,
    write_env_verification_plan,
    write_env_verification_report,
)
from .mujoco_compile import validate_mjcf_loads, write_mjcf
from .operations import execute_operation
from .preview import PreviewError, render_preview
from .schema import EnvSpec3D, env_spec_to_dict
from .studio_view_context import active_screen_requirements, load_studio_view_context


class SceneNotFoundError(KeyError):
    """Raised when a scene session does not exist."""


class SceneSession:
    def __init__(self, *, env_id: str, prompt: str, output_root: Path) -> None:
        self.env_id = env_id
        self.prompt = prompt
        self.output_root = output_root
        self.scene_dir = output_root / env_id
        self.builder = EnvSpec3DBuilder(env_id, description=prompt)
        self.trace_records: list[dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.behavior_turn_id = _behavior_turn_id(self.scene_dir, prompt, self.created_at)
        self._record("create_scene", {"prompt": prompt}, {"status": "success"})
        self._persist_draft_snapshot()

    @classmethod
    def resume(
        cls,
        *,
        env_id: str,
        output_root: Path,
        prompt: str | None = None,
    ) -> "SceneSession":
        scene_dir = output_root / env_id
        paths = artifact_paths(scene_dir)
        spec_data = _read_json(paths.env_spec)
        if spec_data is None:
            raise SceneNotFoundError(f"scene {env_id!r} has no saved env_spec_3d.json")
        session = cls.__new__(cls)
        session.env_id = env_id
        submitted_context = load_studio_view_context(scene_dir)
        submitted_prompt = submitted_context.get("prompt") if isinstance(submitted_context, dict) else None
        session.prompt = str(prompt or submitted_prompt or spec_data.get("description") or "A resumed 3D MuJoCo environment.")
        session.output_root = output_root
        session.scene_dir = scene_dir
        session.builder = EnvSpec3DBuilder.from_spec_dict(spec_data)
        session.trace_records = _read_jsonl(paths.trace)
        session.created_at = datetime.now(timezone.utc).isoformat()
        session.scene_dir.mkdir(parents=True, exist_ok=True)
        session.behavior_turn_id = _behavior_turn_id(session.scene_dir, session.prompt, session.created_at)
        session._record(
            "resume_scene",
            {},
            {
                "status": "success",
                "env_id": env_id,
                "draft_summary": session.builder.inspect_draft(),
            },
        )
        session._persist_draft_snapshot()
        return session

    @property
    def trace_path(self) -> Path:
        return artifact_paths(self.scene_dir).trace

    def apply_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        policy_error = _operation_policy_error(operation, prompt=self.prompt)
        if policy_error:
            result = {
                "success": False,
                "operation": operation,
                "created_ids": [],
                "error": policy_error,
                "draft_summary": self.builder.inspect_draft(),
            }
        else:
            result = execute_operation(self.builder, operation)
        self._record("apply_operation", operation, result)
        if result["success"]:
            self._persist_draft_snapshot()
            if self.builder.game and not load_env_verification_plan(self.scene_dir):
                self._write_game_verification_plan([])
        return {
            "status": "success" if result["success"] else "error",
            "env_id": self.env_id,
            "operation_result": result,
            "draft_summary": self.builder.inspect_draft(),
            "next_action": self._next_action(result["success"]),
        }

    def inspect(self, *, include_spec: bool = False) -> dict[str, Any]:
        response = {
            "status": "success",
            "env_id": self.env_id,
            "prompt": self.prompt,
            "draft_summary": self.builder.inspect_draft(),
        }
        if include_spec:
            response["draft_spec"] = self.builder.to_spec_dict()
        return response

    def render_preview(self, *, camera: str = "overview") -> dict[str, Any]:
        spec = EnvSpec3D.model_validate(self.builder.to_spec_dict())
        draft_xml = self.scene_dir / ".draft_world.xml"
        preview_name = PREVIEW_FILENAMES.get(camera, f"preview_{camera}.png")
        preview_path = self.scene_dir / preview_name
        write_mjcf(spec, draft_xml)
        try:
            render_preview(draft_xml, preview_path, camera=camera)
        except PreviewError as exc:
            result = {"status": "error", "env_id": self.env_id, "error": str(exc), "path": str(preview_path)}
        else:
            result = {
                "status": "success",
                "env_id": self.env_id,
                "camera": camera,
                "path": str(preview_path),
                "url": f"/generated/{self.env_id}/{preview_path.name}?v={preview_path.stat().st_mtime_ns}",
            }
        self._record("render_scene_preview", {"camera": camera}, result)
        return result

    def validate(self) -> dict[str, Any]:
        draft = self.builder.validate_draft()
        spec = EnvSpec3D.model_validate(self.builder.to_spec_dict())
        draft_xml = self.scene_dir / ".draft_world.xml"
        write_mjcf(spec, draft_xml)
        mjcf = validate_mjcf_loads(draft_xml)
        result = {
            "status": "success" if draft["valid"] and mjcf["valid"] else "needs_changes",
            "env_id": self.env_id,
            "draft": draft,
            "mjcf": mjcf,
            "ready_to_finalize": draft["valid"] and mjcf["valid"],
        }
        self._record("validate_scene", {}, result)
        return result

    def define_env_verification_plan(self, *, checks: list[dict[str, Any]]) -> dict[str, Any]:
        screen_context = load_studio_view_context(self.scene_dir)
        _require_submitted_screen_checks(checks, screen_context, prompt=self.prompt)
        plan, path = self._write_game_verification_plan(checks, screen_context=screen_context)
        result = {
            "status": "success",
            "env_id": self.env_id,
            "path": str(path),
            "plan": plan,
            "env_verification": self._env_verification_status(),
            "next_action": "Run run_env_verification after building or editing the scene.",
        }
        self._record("define_env_verification_plan", {"checks": checks}, result)
        return result

    def _write_game_verification_plan(
        self,
        checks: list[dict[str, Any]],
        *,
        screen_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        draft_spec = self.builder.to_spec_dict()
        system_checks = _game_verification_checks(draft_spec)
        system_ids = {check["id"] for check in system_checks}
        merged = [*system_checks, *(check for check in checks if check.get("id") not in system_ids)]
        plan = normalize_env_verification_plan(
            env_id=self.env_id,
            prompt=self.prompt,
            checks=merged,
            operation_count=self.operation_count,
            draft_spec=draft_spec,
            screen_context=screen_context,
        )
        path = write_env_verification_plan(self.scene_dir, plan)
        clear_env_verification_report(self.scene_dir)
        return plan, path

    def run_env_verification(self) -> dict[str, Any]:
        plan = load_env_verification_plan(self.scene_dir)
        if not plan:
            result = {
                "status": "needs_plan",
                "env_id": self.env_id,
                "env_verification": self._env_verification_status(),
                "next_action": "Call define_env_verification_plan with prompt-derived checks first.",
            }
            self._record("run_env_verification", {}, result)
            return result
        draft_spec = self.builder.to_spec_dict()
        draft = self.builder.validate_draft()
        final_spec: dict[str, Any] | None
        if draft["valid"]:
            final_spec = env_spec_to_dict(EnvSpec3D.model_validate(draft_spec))
        else:
            final_spec = None
        report = run_env_verification(
            env_id=self.env_id,
            plan=plan,
            draft_spec=draft_spec,
            final_spec=final_spec,
            operation_count=self.operation_count,
            readiness_errors=draft.get("issues") or [],
        )
        path = write_env_verification_report(self.scene_dir, report)
        result = {
            "status": "needs_changes" if report.get("blocking") else "success",
            "env_id": self.env_id,
            "path": str(path),
            "plan": plan,
            "report": report,
            "env_verification": self._env_verification_status(),
            "next_action": report.get("next_action"),
        }
        self._record("run_env_verification", {}, result)
        return result

    def get_env_verification_report(self) -> dict[str, Any]:
        result = {
            "status": "success",
            "env_id": self.env_id,
            "plan": load_env_verification_plan(self.scene_dir),
            "report": load_env_verification_report(self.scene_dir),
            "env_verification": self._env_verification_status(),
        }
        self._record("get_env_verification_report", {}, result)
        return result

    def define_env_behavior_trials(
        self,
        *,
        trials: list[dict[str, Any]],
        intent_summary: str,
    ) -> dict[str, Any]:
        draft_spec = self.builder.to_spec_dict()
        if not any(obj.get("semantic_type") == "agent" for obj in draft_spec.get("objects") or []):
            result = {
                "status": "not_applicable",
                "env_id": self.env_id,
                "env_behavior_trials": self._behavior_trial_status(),
                "next_action": "Behavior trials are unavailable because this scene has no authored agent.",
            }
            self._record("define_env_behavior_trials", {"trials": trials}, result)
            return result
        plan = normalize_behavior_trial_plan(
            env_id=self.env_id,
            prompt=self.prompt,
            trials=trials,
            operation_count=self.operation_count,
            draft_spec=draft_spec,
            decision="prompt_specific",
            decision_reason=_required_behavior_reason(intent_summary),
            source_turn_id=self.behavior_turn_id,
            intent_summary=intent_summary,
        )
        path = write_behavior_trial_plan(self.scene_dir, plan)
        clear_behavior_trial_report(self.scene_dir)
        result = {
            "status": "success",
            "env_id": self.env_id,
            "path": str(path),
            "plan": plan,
            "env_behavior_trials": self._behavior_trial_status(),
            "next_action": "Finalize the scene, then review and run behavior trials in Studio.",
        }
        self._record("define_env_behavior_trials", {"trials": trials}, result)
        return result

    def preserve_env_behavior_trials(self, *, reason: str) -> dict[str, Any]:
        draft_spec = self.builder.to_spec_dict()
        if not any(obj.get("semantic_type") == "agent" for obj in draft_spec.get("objects") or []):
            return self._behavior_not_applicable("preserve_env_behavior_trials", {"reason": reason})
        existing = load_behavior_trial_plan(self.scene_dir, current_spec=draft_spec)
        if not existing:
            result = {
                "status": "needs_plan",
                "env_id": self.env_id,
                "next_action": "Define prompt-specific agent tests or explicitly choose the default test.",
            }
            self._record("preserve_env_behavior_trials", {"reason": reason}, result)
            return result
        if existing.get("migration_issues"):
            result = {
                "status": "needs_changes",
                "env_id": self.env_id,
                "issues": list(existing.get("migration_issues") or []),
                "next_action": "The existing tests no longer match the scene; define updated prompt-specific tests.",
            }
            self._record("preserve_env_behavior_trials", {"reason": reason}, result)
            return result
        plan = normalize_behavior_trial_plan(
            env_id=self.env_id,
            prompt=self.prompt,
            trials=existing.get("trials") or [],
            operation_count=self.operation_count,
            draft_spec=draft_spec,
            fallback=bool(existing.get("fallback")),
            decision="preserved",
            decision_reason=_required_behavior_reason(reason),
            source_turn_id=self.behavior_turn_id,
            intent_summary=str(existing.get("intent_summary") or _trial_intent_summary(existing)),
        )
        plan["preserved_from_turn_id"] = str(existing.get("source_turn_id") or "")
        plan["preserved_from_prompt_hash"] = str(existing.get("source_prompt_hash") or "")
        path = write_behavior_trial_plan(self.scene_dir, plan)
        clear_behavior_trial_report(self.scene_dir)
        result = {
            "status": "success",
            "env_id": self.env_id,
            "path": str(path),
            "plan": plan,
            "env_behavior_trials": self._behavior_trial_status(),
            "next_action": "Validate and finalize the scene.",
        }
        self._record("preserve_env_behavior_trials", {"reason": reason}, result)
        return result

    def use_default_env_behavior_trial(self, *, reason: str) -> dict[str, Any]:
        draft_spec = self.builder.to_spec_dict()
        if not any(obj.get("semantic_type") == "agent" for obj in draft_spec.get("objects") or []):
            return self._behavior_not_applicable("use_default_env_behavior_trial", {"reason": reason})
        plan = fallback_locomotion_plan(
            env_id=self.env_id,
            prompt=self.prompt,
            operation_count=self.operation_count,
            draft_spec=draft_spec,
            source_turn_id=self.behavior_turn_id,
            decision_reason=_required_behavior_reason(reason),
        )
        if plan is None:  # pragma: no cover - guarded by the authored-agent check
            raise ValueError("default agent test requires an authored agent")
        path = write_behavior_trial_plan(self.scene_dir, plan)
        clear_behavior_trial_report(self.scene_dir)
        result = {
            "status": "success",
            "env_id": self.env_id,
            "path": str(path),
            "plan": plan,
            "env_behavior_trials": self._behavior_trial_status(),
            "next_action": "Validate and finalize the scene.",
        }
        self._record("use_default_env_behavior_trial", {"reason": reason}, result)
        return result

    def _behavior_not_applicable(self, event: str, input_data: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "not_applicable",
            "env_id": self.env_id,
            "env_behavior_trials": self._behavior_trial_status(),
            "next_action": "Agent tests are unavailable because this scene has no authored agent.",
        }
        self._record(event, input_data, result)
        return result

    def get_env_behavior_trial_report(self) -> dict[str, Any]:
        current_spec = self.builder.to_spec_dict()
        plan = load_behavior_trial_plan(self.scene_dir, current_spec=current_spec)
        result = {
            "status": "success",
            "env_id": self.env_id,
            "plan": plan,
            "report": load_behavior_trial_report(self.scene_dir, plan=plan),
            "env_behavior_trials": self._behavior_trial_status(),
        }
        self._record("get_env_behavior_trial_report", {}, result)
        return result

    def finalize(self) -> dict[str, Any]:
        try:
            spec = self.builder.finalize()
        except BuilderValidationError as exc:
            result = {
                "status": "needs_changes",
                "env_id": self.env_id,
                "issues": exc.issues,
                "draft_summary": self.builder.inspect_draft(),
            }
            self._record("finalize_scene", {}, result)
            return result
        verification_block = finalization_block(
            scene_dir=self.scene_dir,
            draft_hash=spec_hash(spec),
            operation_count=self.operation_count,
        )
        if verification_block:
            result = {
                "status": "needs_changes",
                "env_id": self.env_id,
                "env_verification": verification_block,
                "draft_summary": self.builder.inspect_draft(),
            }
            self._record("finalize_scene", {}, result)
            return result
        has_agent = any(obj.semantic_type == "agent" for obj in spec.objects)
        behavior_plan = load_behavior_trial_plan(self.scene_dir, current_spec=spec) if has_agent else None
        behavior_issues = behavior_plan_decision_issues(
            behavior_plan,
            current_spec=spec,
            prompt=self.prompt,
            source_turn_id=self.behavior_turn_id,
        )
        if behavior_issues:
            result = {
                "status": "needs_changes",
                "env_id": self.env_id,
                "env_behavior_trials": {
                    "status": "needs_decision",
                    "label": "Agent tests: decision required",
                    "issues": behavior_issues,
                    "trial_count": len((behavior_plan or {}).get("trials") or []),
                    "next_action": (
                        "Define prompt-specific agent tests, preserve the existing tests, or explicitly choose "
                        "the default test before finalizing."
                    ),
                },
                "draft_summary": self.builder.inspect_draft(),
            }
            self._record("finalize_scene", {}, result)
            return result
        self._record("finalize_scene", {}, {"status": "starting"})
        artifact_result = persist_artifacts(
            spec=spec,
            scene_dir=self.scene_dir,
            trace_records=self.trace_records,
            render=True,
        )
        result = {
            "status": "success",
            "env_id": self.env_id,
            "artifacts": artifact_result["paths"],
            "metadata": artifact_result["metadata"],
            "draft_summary": self.builder.inspect_draft(),
            "env_behavior_trials": self._behavior_trial_status(),
            "env_behavior_trial_plan": behavior_plan,
        }
        return result

    def _record(self, event: str, input_data: dict[str, Any], output_data: dict[str, Any]) -> None:
        record = {
            "event": event,
            "env_id": self.env_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "output": output_data,
        }
        self.trace_records.append(record)
        append_trace_record(self.trace_path, record)

    def _persist_draft_snapshot(self) -> None:
        persist_draft_spec(spec=self.builder.to_spec_dict(), scene_dir=self.scene_dir)

    @staticmethod
    def _next_action(success: bool) -> str:
        if not success:
            return "Correct the operation arguments and retry only the failed edit."
        return "Continue editing, render a preview, validate, or finalize when ready."

    @property
    def operation_count(self) -> int:
        return sum(
            1
            for record in self.trace_records
            if record.get("event") == "apply_operation"
            and isinstance(record.get("output"), dict)
            and record["output"].get("success") is True
        )

    def _env_verification_status(self) -> dict[str, Any]:
        return env_verification_summary(
            self.scene_dir,
            draft_hash=spec_hash(self.builder.to_spec_dict()),
            operation_count=self.operation_count,
        )

    def _behavior_trial_status(self) -> dict[str, Any]:
        return behavior_trial_summary(
            self.scene_dir,
            current_spec=self.builder.to_spec_dict(),
            operation_count=self.operation_count,
        )


class SceneSessionManager:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, SceneSession] = {}

    def create_scene(self, *, env_id: str, prompt: str) -> dict[str, Any]:
        session = SceneSession(env_id=env_id, prompt=prompt, output_root=self.output_root)
        self.sessions[env_id] = session
        return {
            "status": "success",
            "env_id": env_id,
            "prompt": prompt,
            "draft_summary": session.builder.inspect_draft(),
            "next_action": "List the object catalog, then add objects or call a recipe.",
        }

    def resume_scene(self, *, env_id: str, prompt: str | None = None) -> dict[str, Any]:
        session = SceneSession.resume(
            env_id=env_id,
            output_root=self.output_root,
            prompt=prompt,
        )
        self.sessions[env_id] = session
        return {
            "status": "success",
            "env_id": env_id,
            "prompt": session.prompt,
            "draft_summary": session.builder.inspect_draft(),
            "next_action": "Inspect the resumed scene, apply targeted edits, validate, and finalize again.",
        }

    def apply_operation(self, *, env_id: str, operation: dict[str, Any]) -> dict[str, Any]:
        return self._session(env_id).apply_operation(operation)

    def inspect_scene(self, *, env_id: str, include_spec: bool = False) -> dict[str, Any]:
        return self._session(env_id).inspect(include_spec=include_spec)

    def render_scene_preview(self, *, env_id: str, camera: str = "overview") -> dict[str, Any]:
        return self._session(env_id).render_preview(camera=camera)

    def validate_scene(self, *, env_id: str) -> dict[str, Any]:
        return self._session(env_id).validate()

    def define_env_verification_plan(
        self,
        *,
        env_id: str,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._session(env_id).define_env_verification_plan(checks=checks)

    def run_env_verification(self, *, env_id: str) -> dict[str, Any]:
        return self._session(env_id).run_env_verification()

    def get_env_verification_report(self, *, env_id: str) -> dict[str, Any]:
        return self._session(env_id).get_env_verification_report()

    def define_env_behavior_trials(
        self,
        *,
        env_id: str,
        trials: list[dict[str, Any]],
        intent_summary: str,
    ) -> dict[str, Any]:
        return self._session(env_id).define_env_behavior_trials(
            trials=trials,
            intent_summary=intent_summary,
        )

    def preserve_env_behavior_trials(self, *, env_id: str, reason: str) -> dict[str, Any]:
        return self._session(env_id).preserve_env_behavior_trials(reason=reason)

    def use_default_env_behavior_trial(self, *, env_id: str, reason: str) -> dict[str, Any]:
        return self._session(env_id).use_default_env_behavior_trial(reason=reason)

    def get_env_behavior_trial_report(self, *, env_id: str) -> dict[str, Any]:
        return self._session(env_id).get_env_behavior_trial_report()

    def finalize_scene(self, *, env_id: str) -> dict[str, Any]:
        return self._session(env_id).finalize()

    def _session(self, env_id: str) -> SceneSession:
        try:
            return self.sessions[env_id]
        except KeyError as exc:
            raise SceneNotFoundError(f"scene {env_id!r} is not active in this process") from exc


def _behavior_turn_id(scene_dir: Path, prompt: str, created_at: str) -> str:
    context = load_studio_view_context(scene_dir)
    if isinstance(context, dict):
        context_prompt = str(context.get("prompt") or "")
        review_id = str(context.get("review_id") or "").strip()
        if review_id and behavior_prompt_hash(context_prompt) == behavior_prompt_hash(prompt):
            return review_id[:200]
    digest = hashlib.sha256(f"{created_at}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"session-{digest}"


def _required_behavior_reason(value: str) -> str:
    reason = " ".join(str(value or "").split())
    if not reason:
        raise ValueError("an agent-test decision requires a concise reason")
    return reason[:2000]


def _trial_intent_summary(plan: dict[str, Any]) -> str:
    instructions = [
        " ".join(str(trial.get("instruction") or "").split())
        for trial in plan.get("trials") or []
        if isinstance(trial, dict) and str(trial.get("instruction") or "").strip()
    ]
    return " ".join(instructions)[:4000]


def _game_verification_checks(spec: dict[str, Any]) -> list[dict[str, Any]]:
    game = spec.get("game")
    if not isinstance(game, dict):
        return []
    return [
        {
            "id": "system_game_contract",
            "type": "game_contract",
            "severity": "critical",
            "description": "The generated courtyard has a valid, reachable reach-goal game contract.",
        },
        {
            "id": "system_one_agent",
            "type": "object_count",
            "selector": "agent",
            "exact": 1,
            "severity": "critical",
            "description": "The courtyard contains exactly one controllable robot.",
        },
        {
            "id": "system_one_goal",
            "type": "object_count",
            "selector": "goal",
            "exact": 1,
            "severity": "critical",
            "description": "The courtyard contains exactly one charging-pad goal.",
        },
        {
            "id": "system_agent_supported",
            "type": "support_contact",
            "selector": {"id": game.get("agent_id")},
            "surface_selector": {"tag": "walkable"},
            "severity": "critical",
            "description": "The robot starts supported by a walkable courtyard surface.",
        },
        {
            "id": "system_passive_stability",
            "type": "physics_probe",
            "probe": "passive_settles",
            "severity": "critical",
            "description": "The generated MuJoCo scene remains stable when allowed to settle.",
        },
    ]


_GOAL_TERMS_RE = re.compile(
    r"\b(?:goal|charging[ -]?(?:pad|station)|target(?:[ -]?zone)?|destination|finish(?:[ -]?zone)?)\b",
    re.I,
)
_NEGATED_GOAL_RE = re.compile(
    r"\b(?:no|without|never|do\s+not|don['’]?t|dont)\b[^.!?\n]{0,48}"
    r"\b(?:goal|charging[ -]?(?:pad|station)|target(?:[ -]?zone)?|destination|finish(?:[ -]?zone)?)\b",
    re.I,
)
_GENERATED_LEVEL_RE = re.compile(
    r"\b(?:complete\s+(?:generated\s+)?(?:course|level)|generated\s+(?:course|level)|"
    r"obstacle\s+course|new\s+variation|barrier\s+route|slalom|push\s+lane|switch[ -]?gate|"
    r"(?:generate|make|create)\s+(?:me\s+)?(?:a\s+)?(?:full|complete)\s+(?:course|level))\b",
    re.I,
)


def _operation_policy_error(operation: dict[str, Any], *, prompt: str) -> str | None:
    operation_name = str(operation.get("op") or "").strip()
    goal_creating_operations = {"add_goal_zone", "make_courtyard_level", "make_ramp_course", "make_box_goal_scene"}
    if operation_name in goal_creating_operations and not _prompt_requests_goal(prompt):
        return (
            f"{operation_name} would add a goal, but the current user request does not explicitly ask for one. "
            "Keep the scene goal-free and apply only the requested edits."
        )
    if operation_name == "make_courtyard_level" and not _GENERATED_LEVEL_RE.search(prompt):
        return (
            "make_courtyard_level is reserved for an explicit complete generated course or level variation. "
            "Edit the existing four-wall shell with targeted add operations instead."
        )
    return None


def _prompt_requests_goal(prompt: str) -> bool:
    text = " ".join(str(prompt or "").split())
    return bool(_GOAL_TERMS_RE.search(text)) and not bool(_NEGATED_GOAL_RE.search(text))


def _require_submitted_screen_checks(
    checks: list[dict[str, Any]],
    screen_context: dict[str, Any] | None,
    *,
    prompt: str,
) -> None:
    requirements = active_screen_requirements(screen_context, prompt=prompt)
    if not requirements:
        return
    submitted_regions = {
        str(check.get("region") or "").strip().lower().replace("-", "_").replace(" ", "_")
        for check in checks
        if isinstance(check, dict) and str(check.get("type") or "").strip().lower() == "screen_region"
    }
    missing = [requirement["region"] for requirement in requirements if requirement["region"] not in submitted_regions]
    if missing:
        names = ", ".join(region.replace("_", "-") for region in missing)
        raise EnvVerificationError(
            f"This Studio revision requires a screen_region check for {names} using the submitted camera. "
            "Add the check with a subject selector for the edited object; the harness supplies the frozen camera context."
        )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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
