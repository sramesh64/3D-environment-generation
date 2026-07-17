function uniqueIds(values) {
  return [...new Set((values || []).map((value) => String(value || "")).filter(Boolean))];
}

export function buildBehaviorRunView({
  summary = null,
  report = null,
  trialIds = [],
  localRunning = false,
  requestedTrialIds = [],
} = {}) {
  const activeRuns = Array.isArray(summary?.active_runs)
    ? summary.active_runs.filter((run) => run && typeof run === "object")
    : summary?.active_run && typeof summary.active_run === "object"
      ? [summary.active_run]
      : [];
  const activeRun = activeRuns.at(-1) || null;
  const active = Boolean(localRunning || activeRuns.length);
  let activeTrialIds = uniqueIds([
    ...activeRuns.flatMap((run) => run.trial_ids || []),
    ...requestedTrialIds,
  ]);
  if (active && !activeTrialIds.length) activeTrialIds = uniqueIds(trialIds);

  return {
    active,
    activeRuns,
    activeRun,
    activeRunId: String(activeRun?.run_id || ""),
    activeTrialIds,
    runningCount: active ? activeTrialIds.length : 0,
    displayReport: report,
    retainedReport: report,
  };
}

function count(report, key) {
  return Math.max(0, Number(report?.summary?.[key] || 0));
}

function testsLabel(value) {
  return `${value} ${value === 1 ? "test" : "tests"}`;
}

function humanizeIdentifier(value, fallback = "object", capitalize = true) {
  const text = String(value || "")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\s+/g, " ")
    .trim() || fallback;
  const first = capitalize ? text.charAt(0).toUpperCase() : text.charAt(0).toLowerCase();
  return first + text.slice(1);
}

function selectorLabel(selector, fallback, capitalize = true) {
  if (typeof selector === "string") return humanizeIdentifier(selector, fallback, capitalize);
  if (!selector || typeof selector !== "object") return fallback;
  for (const key of ["id", "semantic_type", "object_type", "body_type", "shape", "tag"]) {
    if (selector[key]) return humanizeIdentifier(selector[key], fallback, capitalize);
  }
  return fallback;
}

function isGenericDescription(description, predicateType) {
  const normalized = String(description || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  if (!normalized) return true;
  if (normalized === String(predicateType || "").replaceAll("_", " ")) return true;
  const genericWords = new Set([
    "agent", "axis", "bounds", "check", "condition", "contact", "count", "delta", "displacement",
    "entry", "event", "gain", "grounded", "height", "in", "jump", "mechanism", "objective",
    "object", "overlap", "predicate", "relation", "reset", "settled", "speed", "state", "step",
    "terminal", "test", "value", "zone",
  ]);
  return normalized.split(" ").every((word) => genericWords.has(word));
}

function relationPhrase(relation) {
  return {
    above: "is above",
    below: "is below",
    behind: "is behind",
    far_from: "is far from",
    in_front_of: "is in front of",
    inside: "is inside",
    left_of: "is left of",
    near: "is near",
    on_surface: "is on top of",
    right_of: "is right of",
  }[relation] || `satisfies the ${humanizeIdentifier(relation, "required").toLowerCase()} relation with`;
}

function predicateDescription(check, predicate) {
  const type = String(predicate.type || check?.predicate_type || check?.type || "").toLowerCase();
  const subject = selectorLabel(predicate.subject, "Robot");
  const target = selectorLabel(predicate.target, "target", false);
  if (type === "contact") return `${subject} makes contact with ${target}`;
  if (type === "overlap") return `${subject} enters ${target}`;
  if (type === "relation") return `${subject} ${relationPhrase(String(predicate.relation || ""))} ${target}`;
  if (type === "displacement") return `${subject} moves the required distance`;
  if (type === "axis_delta") {
    const axis = predicate.axis === "z" ? "vertical" : `${predicate.axis || "required"}-axis`;
    return `${subject} changes its ${axis} position by the required amount`;
  }
  if (type === "axis_value") return `${subject} reaches the required ${predicate.axis || "axis"} position`;
  if (type === "speed") return `${subject} has the required ${predicate.component || "linear"} speed`;
  if (type === "settled") return `${subject} comes to rest`;
  if (type === "mechanism_state") {
    const mechanism = humanizeIdentifier(predicate.mechanism_id, "Mechanism");
    return `${mechanism} becomes ${String(predicate.state || "open").replaceAll("_", " ")}`;
  }
  if (type === "jump_count") return "The robot jumps the required number of times";
  if (type === "step_count") return "The run advances through the required simulation steps";
  if (type === "reset_count") return "The run has the required number of resets";
  if (type === "reset_event") {
    const reason = String(predicate.reason || "any").replaceAll("_", " ");
    return reason === "any" ? "A reset occurs" : `A ${reason} reset occurs`;
  }
  if (type === "terminal_event") return `The run records the ${predicate.event || "terminal"} event`;
  if (type === "in_bounds") return `${subject} remains inside the play bounds`;
  if (type === "grounded") return `${subject} is supported by a walkable surface`;
  return "";
}

export function describeBehaviorCheck(check = {}) {
  const predicate = check?.predicate && typeof check.predicate === "object" ? check.predicate : {};
  const predicateType = String(predicate.type || check?.predicate_type || check?.type || "");
  if (check.description && !isGenericDescription(check.description, predicateType)) {
    return String(check.description);
  }
  const generated = predicateDescription(check, predicate);
  if (generated) return `${generated}.`;
  return `${humanizeIdentifier(check.id || predicateType || "objective")}.`;
}

export function describeBehaviorMilestone(frame = {}, objective = {}) {
  const label = String(frame.label || "").trim();
  const completed = label.match(/^Completed:\s*(.*)$/i);
  if (!completed || !isGenericDescription(completed[1], completed[1])) return label;

  const checks = Array.isArray(objective?.checks) ? objective.checks : [];
  const step = Number(frame.step);
  const byStep = Number.isFinite(step)
    ? checks
      .filter((check) => Number.isFinite(Number(check?.first_satisfied_step)))
      .map((check) => ({ check, distance: Math.abs(Number(check.first_satisfied_step) - step) }))
      .sort((left, right) => left.distance - right.distance)[0]
    : null;
  let resolved = byStep && byStep.distance <= 2 ? byStep.check : null;
  if (!resolved && frame.objective_focus?.check_id) {
    resolved = checks.find((check) => String(check?.id || "") === String(frame.objective_focus.check_id));
  }
  if (!resolved && frame.objective_focus?.predicate) {
    resolved = {
      id: frame.objective_focus.check_id,
      description: frame.objective_focus.description,
      predicate: frame.objective_focus.predicate,
    };
  }
  if (!resolved) return label;
  return `Completed: ${describeBehaviorCheck(resolved).replace(/[.\s]+$/, "")}`;
}

function metricNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return String(Number(number.toFixed(2)));
}

function countLabel(value, singular, plural = `${singular}s`) {
  return `${value} ${Number(value) === 1 ? singular : plural}`;
}

export function formatBehaviorMetric(metrics = {}, check = {}) {
  if (Number.isFinite(metrics.distance)) return `${metricNumber(metrics.distance)} m`;
  if (Number.isFinite(metrics.height_gain)) return `${metricNumber(metrics.height_gain)} m`;
  const predicateType = String(
    check?.predicate?.type || check?.predicate_type || check?.type || "",
  ).toLowerCase();
  if (Number.isFinite(metrics.value)) {
    if (["axis_delta", "axis_value", "displacement"].includes(predicateType)) {
      return `${metricNumber(metrics.value)} m`;
    }
    if (predicateType === "jump_count") return countLabel(metrics.value, "jump");
    if (predicateType === "step_count") return countLabel(metrics.value, "step");
    if (predicateType === "reset_count") return countLabel(metrics.value, "reset");
    return metricNumber(metrics.value);
  }
  const countValue = Number.isFinite(metrics.count)
    ? metrics.count
    : Number.isFinite(metrics.transition_count)
      ? metrics.transition_count
      : null;
  if (countValue === null) return "";
  if (predicateType === "contact") return countLabel(countValue, "contact event");
  if (predicateType === "overlap") return countLabel(countValue, "entry", "entries");
  if (predicateType === "relation") return countLabel(countValue, "match", "matches");
  if (predicateType === "mechanism_state") return countLabel(countValue, "activation");
  if (["reset_event", "terminal_event"].includes(predicateType)) return countLabel(countValue, "event");
  return countLabel(countValue, "transition");
}

export function buildBehaviorHeaderView({
  summary = null,
  report = null,
  trialCount = 0,
  active = false,
  runningCount = 0,
} = {}) {
  if (active) {
    const total = Math.max(1, Number(runningCount || trialCount || 1));
    return {
      label: `${testsLabel(total)} running`,
      stats: [],
    };
  }

  const fallbackLabel = String(summary?.label || summary?.status || "Not defined")
    .replace(/^Behavior trials:\s*/i, "")
    .replaceAll("_", " ");
  if (!report) return { label: fallbackLabel, stats: [] };

  const passed = count(report, "passed");
  const inconclusive = count(report, "inconclusive");
  const failed = count(report, "failed");
  const invalidSetup = count(report, "invalid_setup");
  const errors = count(report, "errors");
  const reportedTotal = passed + inconclusive + failed + invalidSetup + errors;
  const total = Math.max(Number(trialCount || 0), reportedTotal);

  if (total > 0 && passed === total) {
    return {
      label: total === 1 ? "1 agent test passed" : `All ${testsLabel(total)} passed`,
      stats: [{ label: "Passed", value: `${passed}/${total}`, tone: "good" }],
    };
  }

  const stats = [];
  if (passed) stats.push({ label: "Passed", value: String(passed), tone: "good" });
  if (inconclusive) stats.push({ label: "Inconclusive", value: String(inconclusive), tone: "warn" });
  if (failed) stats.push({ label: "Failed", value: String(failed), tone: "bad" });
  if (invalidSetup) stats.push({ label: "Invalid setup", value: String(invalidSetup), tone: "bad" });
  if (errors) stats.push({ label: "Errors", value: String(errors), tone: "bad" });
  return { label: fallbackLabel, stats };
}

export function buildBehaviorOutcomeView({ status = "", expectedOutcome = "" } = {}) {
  if (expectedOutcome !== "should_not_succeed") return null;
  if (status === "passed") {
    return {
      tone: "passed",
      label: "The prohibited behavior was not demonstrated during a valid bounded search.",
    };
  }
  if (status === "failed") {
    return {
      tone: "failed",
      label: "The agent demonstrated the prohibited behavior.",
    };
  }
  return {
    tone: "pending",
    label: "The search ended without enough valid evidence to decide this test.",
  };
}
