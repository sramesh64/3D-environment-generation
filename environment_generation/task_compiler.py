"""Read-only Codex compilation of natural-language tasks into typed assertions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env_tasks import TASK_COMPILER_OUTPUT_SCHEMA_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_COMPILER_TIMEOUT_SECONDS = 10 * 60
MAX_COMPILER_ERROR_CHARS = 8_000
MAX_TASK_COMPILER_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class TaskCompilerRepairContext:
    """A rejected compiler result and the validation feedback needed to repair it."""

    attempt: int
    rejected_output: dict[str, Any]
    validation_errors: tuple[str, ...]


def build_task_compiler_prompt(
    *,
    instruction: str,
    spec: dict[str, Any],
    repair_context: TaskCompilerRepairContext | None = None,
) -> str:
    objects = [
        {
            "id": obj.get("id"),
            "semantic_type": obj.get("semantic_type"),
            "body_type": obj.get("body_type"),
            "shape": obj.get("shape"),
            "position": obj.get("position"),
            "size": obj.get("size"),
            "tags": obj.get("tags") or [],
        }
        for obj in spec.get("objects") or []
        if isinstance(obj, dict)
    ]
    context = {
        "env_id": spec.get("id"),
        "task_instruction": instruction,
        "world_size": spec.get("world_size"),
        "play_bounds": (spec.get("game") or {}).get("play_bounds"),
        "objects": objects,
        "mechanisms": spec.get("mechanisms") or [],
    }
    base_prompt = f"""\
You compile one Environment Generation benchmark task into trusted typed trajectory tests.
Return only the JSON required by the supplied output schema. Never emit Python,
JavaScript, source code, expressions, or prose outside that JSON.

Environment and request:
{json.dumps(context, indent=2, ensure_ascii=False)}

Design the smallest set of meaningful tests that fully captures the request.
Use exact object IDs whenever the request identifies a particular object. Every
selector object has id, semantic_type, body_type, shape, and tag fields; set unused
fields to null. Every predicate field is present in the output schema; set fields
that do not apply to null.

For predicates with a subject selector, set subject_quantifier to `all` when every
matched subject must satisfy the predicate (for example every crate must be in the
target), and `any` when one matching subject is sufficient. Prefer exact object IDs
when possible so quantification is unambiguous.

Temporal operators:
- eventually: predicate becomes true at least once.
- at_end: predicate is true in the final recorded state.
- always: predicate stays true for every frame.
- never: predicate is never true.
- sustained: predicate stays true for `frames` consecutive simulation frames.
- count: count false-to-true transitions with min_count/max_count.

Predicate guidance:
- overlap(subject,target): 3D volume overlap, suitable for an agent or movable
  object entering goal, hazard, switch, or target_region sensors.
- relation(subject,target): left/right/front/behind/above/below/near/far/inside/on_surface.
- displacement(subject): final or maximum movement in xy or xyz.
- axis_delta(subject): signed movement along x, y, or z, including maximum,
  minimum, or final displacement from the initial position.
- contact(subject,target): physical contact between selected objects.
- axis_value, speed, settled: numeric or stability conditions on selected objects.
- mechanism_state: activation or gate-open progress for an existing mechanism.
- jump_count, step_count, reset_count, reset_event, terminal_event, in_bounds,
  grounded.

Use mode `all` unless alternatives are genuinely acceptable. Use
ordered_condition_ids only for discrete chronological evidence such as overlap,
contact, relation, mechanism, reset, or terminal events. Never order always, never,
or at_end conditions. Displacement maximum and axis_delta maximum/minimum are
trial-global aggregates: keep them as unordered supporting evidence because they
may become true before an earlier ordered phase.

Important correctness rules:
- Include at least one positive completion condition. A task made only from never
  or always invariants is invalid because waiting would solve it.
- Goal, hazard, switch, and target-region objects are passive semantic sensors.
  Entering one reports an event but does not end or reset a run. Assign meaning
  only through tests: use overlap/eventually when entry is required, and
  overlap/never only when the instruction explicitly requires avoidance.
- Do not add hazard avoidance merely because a hazard exists. Do not use legacy
  terminal_event goal/hazard predicates for new tasks; use overlap predicates.
- Do not require goal entry unless the instruction requires it.
- Do not silently reinterpret goal as a generic object-delivery region. Use an
  authored target_region for object delivery when one exists.
- If the requested target or region does not exist, still reference the intended
  semantic selector. The server will reject it with an actionable missing-object error.
- Avoid redundant tests and avoid asserting exact physics coordinates unless the
  user explicitly asked for them.
- Choose a realistic max_steps between 600 and 12000 for ordinary tasks.
"""
    if repair_context is None:
        return base_prompt
    feedback = [
        {
            "kind": "semantic_validation_error",
            "message": message,
        }
        for message in repair_context.validation_errors
    ]
    return f"""\
{base_prompt}

The previous definition was rejected by the authoritative deterministic validator.
This is repair attempt {repair_context.attempt} of {MAX_TASK_COMPILER_REPAIR_ATTEMPTS}.

Validation feedback:
{json.dumps(feedback, indent=2, ensure_ascii=False)}

Rejected definition:
{json.dumps(repair_context.rejected_output, indent=2, ensure_ascii=False)}

Return a complete corrected replacement, not a patch. Preserve every requirement in
the user's instruction and retain valid conditions from the rejected definition.
Make the smallest semantic correction that resolves all feedback. Do not remove an
ordering requirement merely to silence an ordering error: represent the chronological
milestone with an event-capable condition and keep any required final-state condition
as separate unordered evidence. Do not add requirements that the user did not request.
"""


def build_task_compiler_codex_args(
    *,
    prompt: str,
    output_path: Path,
    model: str = "",
    cwd: Path | None = None,
) -> list[str]:
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
        "--output-schema",
        str(TASK_COMPILER_OUTPUT_SCHEMA_PATH),
        "--output-last-message",
        str(output_path),
    ]
    if cwd is not None:
        args.extend(["-C", str(cwd)])
    if model:
        args.extend(["-m", model])
    args.extend(["--", prompt])
    return args


def run_task_compiler(
    *,
    scene_dir: Path,
    instruction: str,
    spec: dict[str, Any],
    model: str = "",
    repair_context: TaskCompilerRepairContext | None = None,
) -> dict[str, Any]:
    if shutil.which("codex") is None:
        raise RuntimeError("'codex' is not on PATH; install or sign in to the Codex CLI first")
    prompt = build_task_compiler_prompt(
        instruction=instruction,
        spec=spec,
        repair_context=repair_context,
    )
    with tempfile.TemporaryDirectory(prefix="environment-generation-task-compiler-") as temporary:
        output_path = Path(temporary) / "task.json"
        args = build_task_compiler_codex_args(
            prompt=prompt,
            output_path=output_path,
            model=model,
            cwd=scene_dir,
        )
        try:
            completed = subprocess.run(
                args,
                cwd=str(PROJECT_ROOT),
                env=dict(os.environ),
                capture_output=True,
                text=True,
                timeout=TASK_COMPILER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"task compiler exceeded {TASK_COMPILER_TIMEOUT_SECONDS} seconds"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Codex task compiler failed").strip()
            raise RuntimeError(detail[-MAX_COMPILER_ERROR_CHARS:])
        try:
            value = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Codex did not return a valid task definition") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Codex task definition must be a JSON object")
        return value
