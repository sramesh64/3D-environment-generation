from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .behavior_trial import BehaviorTrialSession, MAX_ACTION_FRAMES, MAX_ASSISTED_ACTION_FRAMES
from .catalog import OBJECT_CATALOG_3D
from .env_behavior_trials import (
    MAX_RESETS,
    MAX_STEPS,
    MAX_TRIALS,
    SUPPORTED_EXPECTED_OUTCOMES,
    SUPPORTED_RESET_REASONS,
)
from .env_tasks import (
    SUPPORTED_PREDICATE_TYPES,
    SUPPORTED_RELATIONS as SUPPORTED_ASSERTION_RELATIONS,
    SUPPORTED_TEMPORAL_OPERATORS,
    SUPPORTED_TERMINAL_EVENTS as SUPPORTED_ASSERTION_TERMINAL_EVENTS,
)
from .env_verification import SUPPORTED_CHECK_TYPES, SUPPORTED_PHYSICS_PROBES, SUPPORTED_SPATIAL_RELATIONS
from .operations import OPERATION_DESCRIPTIONS, builder_operation_arg_schemas
from .runtime_config import runtime_env_value
from .session import SceneNotFoundError, SceneSessionManager
from .studio_view_context import SCREEN_REGIONS
from .task_agent import MAX_TASK_ACTION_FRAMES, TASK_AGENT_OBSERVATION_MODE, TaskAgentSession


SERVER_NAME = "environment-generation"
MCP_SERVER_NAME = SERVER_NAME
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"
ENV_ID_PATTERN = "^[A-Za-z0-9_-]+$"


WORKFLOW_GUIDE = """\
Build 3D MuJoCo scenes incrementally and wait for every tool result.
1. Call create_scene once for a new env, or resume_scene with the latest user request for an existing env revision.
2. Call list_object_catalog before choosing objects.
3. A new Studio scene already contains its ground and four boundary walls. Inspect that shell, then add only objects or structures requested by the user. Do not replace it with a recipe or add decorative gameplay objects merely to make the scene feel complete.
4. Never add an agent, goal, hazard, switch, gate, route barrier, or interior prop unless the request calls for it. In particular, do not add a goal or charging pad unless the current user request explicitly asks for a goal, charging pad, target, or destination. An unqualified box or crate is pushable unless the user asks for an anchored or structural box.
5. Use make_courtyard_level only when the user explicitly requests a complete generated course, level variation, or one of its named families. It is never the default first operation.
6. Agent and goal objects are independent. When the authored scene contains exactly one explicitly requested agent and one explicitly requested goal, call configure_reach_goal_game to activate reach-goal play. Do not invent a missing endpoint for the contract.
7. Use z-up coordinates: x right, y forward, z up. Builder operation dimensions are full dimensions, not MuJoCo half-extents. Author and revise ramps only with add_ramp/set_ramp_geometry: x/y/z are the low walkable endpoint, yaw points uphill, length is horizontal run, rise is vertical gain, and thickness is collider thickness.
8. For revisions, inspect the current scene before editing and make targeted changes. Preserve an existing game contract unless the user removes one of its referenced objects.
9. Wall segments that form one logical barrier must use the same appearance. New walls default to solid stone; choose fence or hedge only when the user asks for that presentation. Render preview images while editing when they help inspect the authored scene.
10. Define deterministic env verification checks for concrete prompt requirements, run them after major edits, and repair critical failures. Use ramp_connection whenever a ramp must join named lower and upper surfaces; do not replace it with a loose center-distance check. For a Studio request using screen-relative regions such as bottom-left, use the attached submitted-view image and screen_space.regions anchor, then include a screen_region check for the edited object. For object-relative phrases such as "left of the box," use screen_relation rather than the world-axis spatial_relation. The harness freezes the submitted camera and will reject plans that omit a required absolute region check.
11. Before finalizing any scene with an authored agent, make exactly one current-turn agent-test decision. If the latest request introduces or changes a concrete affordance, call define_env_behavior_trials and put a test that directly covers the latest request first; its typed checks must represent the essential observable stages, with at most one still-relevant regression test second. If an edit can affect an existing affordance without introducing a new one, redefine the still-relevant prior tests against the current draft. Call preserve_env_behavior_trials only when the edit cannot affect behavior or test semantics. Use use_default_env_behavior_trial only when the current conversation contains no concrete behavioral requirement. Never let the generic fallback replace a prompt-specific request. Use reusable typed trajectory assertions with explicit subjects and targets: for example, agent entering a goal is overlap(subject=agent, target=goal), while delivering a crate is overlap(subject=crate, target=region). Goal, hazard, switch, and target-region sensors are passive: entering them emits evidence but has no built-in success, failure, or reset meaning. Express required entry with an overlap objective and forbidden entry with an explicit never constraint; do not infer hazard avoidance merely because a hazard exists. Give every check a concise user-facing description that names the subject, observable action or relation, and target; never use only a predicate type such as "contact" or "relation". Combine assertions with temporal operators. Use ordered check ids only for discrete chronological evidence such as overlap, contact, relation, or mechanism events. Keep always/never/at_end checks and trial-global displacement/axis-delta maximum or minimum metrics unordered. Never misuse an agent-specific shorthand for object delivery. For should_not_succeed, describe the prohibited counterexample positively. Put only genuine attempt rules in constraints.
12. Validate before finalizing. Every scene needs valid MJCF and passing critical environment checks; a game contract is required only when reach-goal gameplay has been explicitly authored.
Studio runs a separate post-finalization multi-view visual review. Do not define or run VLM/visual-review checks in the main environment-generation turn. Behavior trials are generation-time affordance probes, not user tasks, and Studio runs them only after user approval.
env_spec_3d.json is the portable source of truth; world.xml is derived MJCF.
"""

BEHAVIOR_WORKFLOW_GUIDE = """\
Control the bound immutable Environment Generation behavior trial. Start the trial, inspect the
first-person frame and telemetry, and use controller actions in short batches. For
long collision-aware ground approaches, the existing action tool can follow the
advisory route while preserving exact low-level actions for replay. It can target a
specific known scene object for multi-stage routes such as reaching a switch before
the final objective. For manipulation,
follow interaction_guidance: approach its predicate-specific, shape-aware staging
point behind the subject, align the subject with its desired center, then push
manually in short batches. Geometry is advisory; only objective checks establish
completion. You may reset within the
reported budget. Semantic zones emit evidence but never terminate by themselves;
only typed objectives, typed constraints, budgets, or hard out-of-bounds safety can
end the trial. Reset and try another strategy when the trial is not yet satisfied. Stop when the objective is visibly and
programmatically satisfied, or when distinct attempts show that it was not
demonstrated. Do not claim that a route is impossible merely because you did not
find it. No scene-editing tools are available in this mode.
"""

TASK_WORKFLOW_GUIDE = """\
Control the bound immutable Environment Generation benchmark task. Start the run, inspect the
first-person frame, then act in exact-frame batches. Use longer batches on clear
ground and short batches for turns, contacts, and narrow spaces. The task interface is
visual-first and exposes only human-like anonymous collision, grounded/airborne,
and zone-entry cues alongside recent action history. It does not
expose coordinates, semantic object labels, zone identities, target bearings,
mechanism state, maps, a global preview, or hidden deterministic test progress.
Movement actions stop early when a solid collision blocks the requested direction,
so inspect the returned frame before continuing. You may reset within the reported
budget. Follow the task instruction and infer progress from visible evidence. Stop
when the episode reports a terminal outcome or bounded attempts are exhausted. No
scene, task, test, shell, browsing, or file-editing tools are available in this mode.
"""


class EnvironmentGenerationMCPServer:
    def __init__(self, output_root: str | Path | None = None) -> None:
        default_root = Path(__file__).resolve().parent.parent / "generated"
        configured = output_root or runtime_env_value("OUTPUT_ROOT") or default_root
        self.sessions = SceneSessionManager(configured)
        self.behavior_session = (
            BehaviorTrialSession.from_environment()
            if runtime_env_value("BEHAVIOR_SCENE_DIR")
            else None
        )
        self.task_session = (
            TaskAgentSession.from_environment()
            if runtime_env_value("TASK_SCENE_DIR")
            else None
        )
        if self.behavior_session is not None and self.task_session is not None:
            raise ValueError("behavior and task child modes cannot be active together")
        self._operation_schemas = builder_operation_arg_schemas()
        self._tools = self._build_tools()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._success(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}, "prompts": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": (
                        BEHAVIOR_WORKFLOW_GUIDE
                        if self.behavior_session
                        else TASK_WORKFLOW_GUIDE
                        if self.task_session
                        else WORKFLOW_GUIDE
                    ),
                },
            )
        if method == "ping":
            return self._success(request_id, {})
        if method == "tools/list":
            return self._success(request_id, {"tools": self._tools})
        if method == "tools/call":
            return self._call_tool(request_id, params)
        if method == "resources/list":
            return self._success(request_id, {"resources": []})
        if method == "prompts/list":
            return self._success(
                request_id,
                {
                    "prompts": [
                        {
                            "name": "build_3d_environment",
                            "description": "Incrementally build, inspect, render, validate, and finalize a MuJoCo environment.",
                            "arguments": [
                                {"name": "request", "description": "User's environment request", "required": True},
                                {"name": "env_id", "description": "Stable environment ID", "required": True},
                            ],
                        }
                    ]
                },
            )
        if method == "prompts/get":
            if params.get("name") != "build_3d_environment":
                return self._error(request_id, -32602, "unknown prompt")
            arguments = params.get("arguments") or {}
            return self._success(
                request_id,
                {
                    "description": "Environment Generation for MuJoCo",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": (
                                    f"Build 3D environment {arguments.get('env_id', 'studio_env_001')!r}: "
                                    f"{arguments.get('request', '')}\n\n{WORKFLOW_GUIDE}"
                                ),
                            },
                        }
                    ],
                },
            )
        return self._error(request_id, -32601, f"Method not found: {method}")

    def _call_tool(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = dict(params.get("arguments") or {})
        try:
            result = self._dispatch_tool(name, arguments)
        except SceneNotFoundError as exc:
            return self._error(request_id, -32004, str(exc))
        except Exception as exc:
            return self._error(request_id, -32000, str(exc))
        return self._success(request_id, {"content": self._content_for_result(result)})

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.task_session is not None:
            tools: dict[str, Callable[..., dict[str, Any]]] = {
                "start_task_run": self.task_session.start,
                "observe_task_run": self.task_session.observe,
                "act_task_run": self.task_session.act,
                "reset_task_run": self.task_session.reset,
            }
            if name == "stop_task_run":
                return _compact_task_result(self.task_session.stop())
            if name not in tools:
                raise KeyError(f"unknown task tool {name!r}")
            return tools[name](**arguments)
        if self.behavior_session is not None:
            tools: dict[str, Callable[..., dict[str, Any]]] = {
                "start_behavior_trial": self.behavior_session.start,
                "observe_behavior_trial": self.behavior_session.observe,
                "act_behavior_trial": self.behavior_session.act,
                "reset_behavior_trial": self.behavior_session.reset,
            }
            if name == "stop_behavior_trial":
                return _compact_behavior_result(self.behavior_session.stop())
            if name not in tools:
                raise KeyError(f"unknown behavior tool {name!r}")
            return tools[name](**arguments)
        tools: dict[str, Callable[..., dict[str, Any]]] = {
            "create_scene": self.sessions.create_scene,
            "resume_scene": self.sessions.resume_scene,
            "list_object_catalog": lambda: {
                "status": "success",
                "catalog": OBJECT_CATALOG_3D,
                "operations": [
                    {
                        "name": name,
                        "description": OPERATION_DESCRIPTIONS[name],
                        "arguments": schema,
                    }
                    for name, schema in self._operation_schemas.items()
                ],
            },
            "apply_operation": self.sessions.apply_operation,
            "inspect_scene": self.sessions.inspect_scene,
            "render_scene_preview": self.sessions.render_scene_preview,
            "validate_scene": self.sessions.validate_scene,
            "define_env_verification_plan": self.sessions.define_env_verification_plan,
            "run_env_verification": self.sessions.run_env_verification,
            "get_env_verification_report": self.sessions.get_env_verification_report,
            "define_env_behavior_trials": self.sessions.define_env_behavior_trials,
            "preserve_env_behavior_trials": self.sessions.preserve_env_behavior_trials,
            "use_default_env_behavior_trial": self.sessions.use_default_env_behavior_trial,
            "get_env_behavior_trial_report": self.sessions.get_env_behavior_trial_report,
            "finalize_scene": self.sessions.finalize_scene,
        }
        if name not in tools:
            raise KeyError(f"unknown tool {name!r}")
        return tools[name](**arguments)

    def _build_tools(self) -> list[dict[str, Any]]:
        if self.task_session is not None:
            return _task_agent_tools()
        if self.behavior_session is not None:
            return _behavior_navigation_tools()
        return [
            {
                "name": "create_scene",
                "description": "Start a new persistent 3D scene-editing session.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id", "prompt"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "prompt": {"type": "string", "minLength": 1},
                    },
                },
            },
            {
                "name": "resume_scene",
                "description": "Reopen a saved finalized 3D scene from env_spec_3d.json for conversational revisions.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "prompt": {
                            "type": "string",
                            "description": "The latest user revision request for plan provenance.",
                        },
                    },
                },
            },
            {
                "name": "list_object_catalog",
                "description": "Return supported 3D primitives, recipes, and coordinate conventions.",
                "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
            },
            {
                "name": "apply_operation",
                "description": "Apply one semantic scene operation to a 3D draft.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id", "operation"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "operation": _builder_operation_schema(self._operation_schemas),
                    },
                },
            },
            {
                "name": "inspect_scene",
                "description": "Inspect current object inventory and draft readiness.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "include_spec": {"type": "boolean", "default": False},
                    },
                },
            },
            {
                "name": "render_scene_preview",
                "description": "Render the current draft with one of the fixed MuJoCo cameras.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "camera": {"type": "string", "enum": ["overview", "agent", "goal"], "default": "overview"},
                    },
                },
            },
            {
                "name": "validate_scene",
                "description": "Validate draft completeness and MuJoCo MJCF loadability.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string", "pattern": ENV_ID_PATTERN}},
                },
            },
            {
                "name": "define_env_verification_plan",
                "description": "Submit prompt-derived deterministic generation checks for this 3D environment.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id", "checks"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "checks": {
                            "type": "array",
                            "minItems": 1,
                            "description": (
                                "Supported types: "
                                + ", ".join(sorted(SUPPORTED_CHECK_TYPES))
                                + ". Select objects with a string like 'agent' or an object containing id, "
                                "semantic_type/object_type, body_type, shape, or tag."
                            ),
                            "items": _env_verification_check_schema(),
                        },
                    },
                },
            },
            {
                "name": "run_env_verification",
                "description": "Evaluate the current draft against prompt-derived deterministic env checks.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string", "pattern": ENV_ID_PATTERN}},
                },
            },
            {
                "name": "get_env_verification_report",
                "description": "Return the latest deterministic env verification plan and report.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string", "pattern": ENV_ID_PATTERN}},
                },
            },
            {
                "name": "define_env_behavior_trials",
                "description": (
                    "Define one or two short prompt-derived affordance trials for an authored agent. "
                    "Objectives describe the behavior to demonstrate, including the prohibited counterexample "
                    "for should_not_succeed. Put only genuine attempt rules in constraints; do not make the "
                    "counterexample impossible by definition. Use composable typed trajectory assertions with "
                    "explicit subjects, targets, temporal operators, and ordering; do not add full tasks or "
                    "generated verifier code."
                ),
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id", "intent_summary", "trials"],
                    "properties": {
                        "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
                        "intent_summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 4000,
                            "description": (
                                "Concise account of the latest prompt-specific behavior covered by these tests. "
                                "The first test must cover this intent rather than a generic game objective."
                            ),
                        },
                        "trials": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_TRIALS,
                            "items": _env_behavior_trial_schema(),
                        },
                    },
                },
            },
            {
                "name": "preserve_env_behavior_trials",
                "description": (
                    "Explicitly preserve and revalidate the existing agent tests for the current scene draft. "
                    "Use only when the latest edit cannot affect behavior or test semantics."
                ),
                "inputSchema": _behavior_decision_schema(),
            },
            {
                "name": "use_default_env_behavior_trial",
                "description": (
                    "Explicitly choose the generic locomotion or reach-goal agent test for this turn. "
                    "Use only when the conversation contains no concrete prompt-specific affordance."
                ),
                "inputSchema": _behavior_decision_schema(),
            },
            {
                "name": "get_env_behavior_trial_report",
                "description": "Return the current behavior-trial plan and latest code-scored report.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string", "pattern": ENV_ID_PATTERN}},
                },
            },
            {
                "name": "finalize_scene",
                "description": (
                    "Finalize the scene into env_spec_3d.json, world.xml, previews, trace, and metadata. "
                    "Scenes with an agent require an explicit current-turn agent-test decision first."
                ),
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string", "pattern": ENV_ID_PATTERN}},
                },
            },
        ]

    def _content_for_result(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
        if result.get("status") != "success":
            return content
        paths: list[str] = []
        # Task conversations already retain earlier tool images. Reattaching the
        # same three frames on every action triples vision work without adding
        # evidence; attach only the current view while preserving frame metadata.
        if result.get("observation_mode") != TASK_AGENT_OBSERVATION_MODE:
            for frame in result.get("recent_frames") or []:
                path = frame.get("path") if isinstance(frame, dict) else None
                if isinstance(path, str) and path.endswith(".png") and path not in paths:
                    paths.append(path)
        path = result.get("path")
        if isinstance(path, str) and path.endswith(".png") and path not in paths:
            paths.append(path)
        for path in paths:
            try:
                data = Path(path).read_bytes()
            except OSError:
                continue
            content.append({"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/png"})
        return content

    @staticmethod
    def _success(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _env_verification_check_schema() -> dict[str, Any]:
    selector_schema = {
        "anyOf": [
            {"type": "string"},
            {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "id": {"type": "string"},
                    "semantic_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "object_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "body_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "shape": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "tag": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                },
            },
        ]
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["type"],
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": sorted(SUPPORTED_CHECK_TYPES)},
            "severity": {"type": "string", "enum": ["critical", "advisory"], "default": "critical"},
            "description": {"type": "string"},
            "selector": selector_schema,
            "subject": selector_schema,
            "target": {"anyOf": [selector_schema, {"type": "array", "items": selector_schema}]},
            "targets": {"type": "array", "items": selector_schema, "minItems": 2, "maxItems": 2},
            "relation": {"type": "string", "enum": sorted(SUPPORTED_SPATIAL_RELATIONS)},
            "region": {"type": "string", "enum": sorted(SCREEN_REGIONS)},
            "margin_uv": {"type": "number", "minimum": 0, "maximum": 1},
            "exact": {"type": "integer", "minimum": 0},
            "min": {"type": "integer", "minimum": 0},
            "max": {"type": "integer", "minimum": 0},
            "distance": {"type": "number", "exclusiveMinimum": 0},
            "max_distance": {"type": "number", "exclusiveMinimum": 0},
            "min_distance": {"type": "number", "exclusiveMinimum": 0},
            "surface_selector": selector_schema,
            "ramp": selector_schema,
            "low_surface": selector_schema,
            "high_surface": selector_schema,
            "max_gap": {"type": "number", "minimum": 0},
            "max_horizontal_gap": {"type": "number", "minimum": 0},
            "max_vertical_gap": {"type": "number", "minimum": 0},
            "probe": {"type": "string", "enum": sorted(SUPPORTED_PHYSICS_PROBES)},
            "object_id": {"type": "string"},
        },
    }


def _builder_operation_schema(schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["op", "args"],
        "properties": {
            "op": {"type": "string", "enum": list(schemas)},
            "args": {
                "type": "object",
                "description": "Arguments matching the selected operation-specific schema below.",
            },
        },
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "args"],
                "properties": {
                    "op": {
                        "const": name,
                        "description": OPERATION_DESCRIPTIONS[name],
                    },
                    "args": schema,
                },
            }
            for name, schema in schemas.items()
        ]
    }


def _selector_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string"},
            {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "id": {"type": "string"},
                    "semantic_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "object_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "body_type": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "shape": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "tag": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                },
            },
        ]
    }


def _behavior_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["env_id", "reason"],
        "properties": {
            "env_id": {"type": "string", "pattern": ENV_ID_PATTERN},
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Why preserving existing tests or selecting the default is correct for this turn.",
            },
        },
    }


def _env_behavior_trial_schema() -> dict[str, Any]:
    selector = _selector_schema()
    predicate = {
        "type": "object",
        "additionalProperties": True,
        "required": ["type"],
        "properties": {
            "type": {"type": "string", "enum": sorted(SUPPORTED_PREDICATE_TYPES)},
            "subject": selector,
            "target": selector,
            "subject_quantifier": {"type": "string", "enum": ["any", "all"], "default": "any"},
            "relation": {
                "type": "string",
                "enum": sorted(SUPPORTED_ASSERTION_RELATIONS),
            },
            "space": {"type": "string", "enum": ["xy", "xyz"], "default": "xy"},
            "axis": {"type": "string", "enum": ["x", "y", "z"]},
            "component": {"type": "string", "enum": ["linear", "angular"]},
            "metric": {"type": "string", "enum": ["maximum", "minimum", "final"], "default": "maximum"},
            "min_value": {"type": "number"},
            "max_value": {"type": "number"},
            "margin": {"type": "number"},
            "min_distance": {"type": "number", "minimum": 0},
            "max_distance": {"type": "number", "minimum": 0},
            "max_linear_speed": {"type": "number", "minimum": 0},
            "max_angular_speed": {"type": "number", "minimum": 0},
            "mechanism_id": {"type": "string"},
            "state": {"type": "string", "enum": ["active", "open"]},
            "min_progress": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "event": {
                "type": "string",
                "enum": sorted(SUPPORTED_ASSERTION_TERMINAL_EVENTS),
            },
            "reason": {"type": "string", "enum": sorted(SUPPORTED_RESET_REASONS)},
        },
    }
    check = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "description", "predicate"],
        "properties": {
            "id": {"type": "string"},
            "description": {
                "type": "string",
                "minLength": 8,
                "maxLength": 600,
                "description": (
                    "Plain-language outcome naming the subject, observable action or relation, and target; "
                    "do not use only a predicate type such as 'contact' or 'relation'."
                ),
            },
            "temporal": {
                "type": "string",
                "enum": sorted(SUPPORTED_TEMPORAL_OPERATORS),
                "default": "eventually",
            },
            "frames": {"type": "integer", "minimum": 1},
            "min_count": {"type": "integer", "minimum": 0},
            "max_count": {"type": "integer", "minimum": 0},
            "predicate": predicate,
        },
    }
    objective_group = {
        "type": "object",
        "additionalProperties": False,
        "required": ["checks"],
        "properties": {
            "mode": {"type": "string", "enum": ["all", "any"], "default": "all"},
            "checks": {"type": "array", "minItems": 1, "items": check},
            "ordered_check_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Chronological IDs for discrete event-like conditions only. Keep always, never, at_end, "
                    "and displacement/axis_delta maximum or minimum aggregate conditions unordered."
                ),
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["instruction", "objective"],
        "properties": {
            "id": {"type": "string"},
            "instruction": {"type": "string", "minLength": 1},
            "expected_outcome": {
                "type": "string",
                "enum": sorted(SUPPORTED_EXPECTED_OUTCOMES),
                "description": (
                    "Whether demonstrating objective is expected. For should_not_succeed, objective is the "
                    "prohibited counterexample, never the expected safe condition."
                ),
            },
            "severity": {"type": "string", "enum": ["critical", "advisory"]},
            "max_steps": {
                "type": "integer",
                "minimum": 60,
                "maximum": MAX_STEPS,
                "description": "Initial per-attempt budget; runtime preflight may expand it to reach the target.",
            },
            "max_resets": {"type": "integer", "minimum": 0, "maximum": MAX_RESETS},
            "objective": {
                **objective_group,
                "description": "Typed behavior that the rollout is trying to demonstrate.",
            },
            "constraints": {
                **objective_group,
                "description": (
                    "Optional restrictions that must hold in the same attempt, such as no jumping or no target "
                    "movement. Constraints do not describe the expected safe outcome."
                ),
            },
            "allow_target_motion": {
                "type": "boolean",
                "default": False,
                "description": "Allow an intentionally moving dynamic target to bypass stability preflight.",
            },
        },
    }


def _behavior_navigation_tools() -> list[dict[str, Any]]:
    empty = {"type": "object", "additionalProperties": False, "properties": {}}
    return [
        {"name": "start_behavior_trial", "description": "Start the bound immutable behavior trial.", "inputSchema": empty},
        {"name": "observe_behavior_trial", "description": "Observe first-person evidence and compact simulator telemetry.", "inputSchema": empty},
        {
            "name": "act_behavior_trial",
            "description": (
                "Apply a manual controller action, or invoke bounded advisory ground-route following. "
                "Assisted movement records exact low-level controller segments for authoritative replay."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forward": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "right": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "look_x": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "look_y": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "jump": {"type": "boolean", "default": False},
                    "assist": {
                        "type": "string",
                        "enum": ["none", "ground_route"],
                        "default": "none",
                        "description": (
                            "Use ground_route for objective-conditioned collision-aware approach only. "
                            "Use manual actions for jumping, pushing, searching, and final interactions."
                        ),
                    },
                    "target_id": {
                        "type": "string",
                        "description": (
                            "Optional scene-object destination for ground_route assistance. "
                            "Use this for an intermediate target such as a linked switch; "
                            "omit it to approach the current objective target."
                        ),
                    },
                    "frames": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_ASSISTED_ACTION_FRAMES,
                        "default": 12,
                        "description": (
                            f"Manual actions support at most {MAX_ACTION_FRAMES} frames; ground_route "
                            f"assistance supports at most {MAX_ASSISTED_ACTION_FRAMES}."
                        ),
                    },
                },
            },
        },
        {"name": "reset_behavior_trial", "description": "Reset for another attempt within the trial budget.", "inputSchema": empty},
        {"name": "stop_behavior_trial", "description": "Stop and return the current code-scored outcome.", "inputSchema": empty},
    ]


def _task_agent_tools() -> list[dict[str, Any]]:
    empty = {"type": "object", "additionalProperties": False, "properties": {}}
    return [
        {"name": "start_task_run", "description": "Start the bound immutable benchmark task.", "inputSchema": empty},
        {
            "name": "observe_task_run",
            "description": (
                "Observe the current first-person image, recent actions, anonymous "
                "collision/ground/zone-entry cues, and public episode budget. Earlier "
                "views remain available in the conversation."
            ),
            "inputSchema": empty,
        },
        {
            "name": "act_task_run",
            "description": (
                "Apply one controller action for up to the requested MuJoCo frames; solid "
                "collisions may end movement early so the next view is not skipped."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "forward": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "right": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "look_x": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "look_y": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
                    "jump": {"type": "boolean", "default": False},
                    "frames": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TASK_ACTION_FRAMES,
                        "default": 24,
                        "description": (
                            "Use 36-90 frames for deliberate travel over clear ground and "
                            "6-24 for turns, contacts, hazards, and precise placement."
                        ),
                    },
                },
            },
        },
        {"name": "reset_task_run", "description": "Reset for another attempt within the task budget.", "inputSchema": empty},
        {"name": "stop_task_run", "description": "Stop and return the current deterministic test outcome.", "inputSchema": empty},
    ]


def _compact_behavior_result(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {"actions", "events", "attempts", "trajectory", "final_state"}
    compact = {key: value for key, value in result.items() if key not in omitted}
    compact["action_count"] = len(result.get("actions") or [])
    compact["event_count"] = len(result.get("events") or [])
    compact["trajectory_frame_count"] = len(result.get("trajectory") or [])
    return compact


def _compact_task_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result.get("task_id"),
        "observation_mode": result.get("observation_mode"),
        "status": result.get("status"),
        "passed": bool(result.get("passed")),
        "steps_used": int(result.get("steps_used") or 0),
        "reset_count": int(result.get("reset_count") or 0),
        "action_count": len(result.get("actions") or []),
        "trajectory_frame_count": len(result.get("trajectory") or []),
    }


def main() -> None:
    server = EnvironmentGenerationMCPServer()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = server.handle_request(request)
            except Exception as exc:
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
            if response is not None:
                print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), flush=True)
    finally:
        if server.behavior_session is not None:
            server.behavior_session.close()
        if server.task_session is not None:
            server.task_session.close()


if __name__ == "__main__":
    main()
