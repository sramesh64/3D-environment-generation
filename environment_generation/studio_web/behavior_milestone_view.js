const DEFAULT_VIEW = {
  target: [0, 0, 0.7],
  distance: 8,
  azimuth: -42,
  elevation: 46,
  panX: 0,
  panY: 0,
};

export function nearestTrajectoryFrame(frames, targetStep, targetAttempt = null) {
  const index = nearestTrajectoryFrameIndex(frames, targetStep, targetAttempt);
  return index >= 0 ? frames[index] : null;
}

export function nearestTrajectoryFrameIndex(frames, targetStep, targetAttempt = null) {
  const values = Array.isArray(frames)
    ? frames.map((frame, index) => ({ frame, index })).filter(({ frame }) => Boolean(frame))
    : [];
  if (!values.length) return -1;
  const attempt = Number(targetAttempt);
  const hasAttempt = targetAttempt !== null
    && targetAttempt !== undefined
    && targetAttempt !== ""
    && Number.isFinite(attempt);
  const matchingAttempt = hasAttempt
    ? values.filter(({ frame }) => Number(frame?.attempt) === attempt)
    : [];
  const candidates = matchingAttempt.length ? matchingAttempt : values;
  const desired = finiteNumber(targetStep, 0);
  return candidates.reduce((best, candidate) => {
    const bestDistance = Math.abs(finiteNumber(best.frame?.total_step, 0) - desired);
    const distance = Math.abs(finiteNumber(candidate.frame?.total_step, 0) - desired);
    return distance < bestDistance ? candidate : best;
  }, candidates[0]).index;
}

export function behaviorReplayFrameState(frame, selection = null) {
  const source = frame && typeof frame === "object" ? frame : {};
  const selected = selection && typeof selection === "object" ? selection : {};
  let objects = Array.isArray(selected.objects)
    ? selected.objects
    : Array.isArray(source.objects)
      ? source.objects
      : [];
  const agentId = String(selected.agent?.id || "");
  const agentPosition = finiteVector(selected.agent?.position);
  if (!Array.isArray(selected.objects) && agentId && agentPosition) {
    objects = objects.map((object) => (
      String(object?.id || "") === agentId
        ? { ...object, position: agentPosition.slice() }
        : object
    ));
  }
  return {
    objects,
    mechanisms: Array.isArray(selected.mechanisms)
      ? selected.mechanisms
      : Array.isArray(source.mechanisms)
        ? source.mechanisms
        : [],
    view: selected.view && typeof selected.view === "object" ? selected.view : null,
  };
}

export function inheritBehaviorMilestoneFocus(frames) {
  const values = Array.isArray(frames) ? frames.filter(Boolean) : [];
  const focusIdsByCheck = new Map();
  return values.map((source) => {
    const frame = { ...source };
    const focus = frame.objective_focus || {};
    const currentFocusIds = unique([
      ...(focus.subject_ids || []),
      ...(focus.target_ids || []),
    ].map(String).filter(Boolean));
    const currentCheckId = String(focus.check_id || "");
    if (currentCheckId && currentFocusIds.length) {
      focusIdsByCheck.set(currentCheckId, currentFocusIds);
    }

    let focusIds = unique((frame.focus_ids || []).map(String).filter(Boolean));
    const eventIds = unique(
      (frame.events || []).map((event) => String(event?.object_id || "")).filter(Boolean),
    );
    if (!focusIds.length && frame.kind === "event" && eventIds.length) focusIds = eventIds;
    if (!focusIds.length && ["objective", "final"].includes(String(frame.kind || ""))) {
      const completedCheck = completedObjectiveCheck(frame);
      if (completedCheck) focusIds = focusIdsByCheck.get(String(completedCheck.id || "")) || [];
    }
    if (!focusIds.length) {
      const navigationId = String(frame.navigation?.primary_target_id || "");
      focusIds = unique([
        ...currentFocusIds,
        ...(navigationId ? [navigationId] : []),
        ...eventIds,
      ]);
    }
    frame.focus_ids = focusIds;
    return frame;
  });
}

export function buildBehaviorMilestoneView({
  visualScene = null,
  evidenceFrame = null,
  trajectoryFrame = null,
  mechanisms = [],
} = {}) {
  const scene = visualScene && typeof visualScene === "object" ? visualScene : {};
  const evidence = evidenceFrame && typeof evidenceFrame === "object" ? evidenceFrame : {};
  const trajectory = trajectoryFrame && typeof trajectoryFrame === "object" ? trajectoryFrame : {};
  const sceneObjects = Array.isArray(scene.objects) ? scene.objects : [];
  const positions = new Map();
  const sizes = new Map();

  for (const object of sceneObjects) {
    const id = String(object?.source_id || object?.id || "");
    const position = finiteVector(object?.position);
    if (!id || !position) continue;
    positions.set(id, position);
    sizes.set(id, finiteSize(object?.size));
  }
  for (const object of trajectory.objects || []) {
    const id = String(object?.id || "");
    const position = finiteVector(object?.position);
    if (id && position) positions.set(id, position);
  }

  const authoredAgent = sceneObjects.find((object) => String(object?.semantic_type || "") === "agent");
  const agentId = String(evidence.agent?.id || authoredAgent?.source_id || authoredAgent?.id || "");
  const evidenceAgentPosition = finiteVector(evidence.agent?.position);
  if (agentId && evidenceAgentPosition) positions.set(agentId, evidenceAgentPosition);

  const focusIds = expandMechanismFocusIds(
    collectFocusIds(evidence, positions),
    evidence,
    mechanisms,
    positions,
  ).filter((id) => id !== agentId);
  const anchorIds = unique([agentId, ...focusIds]).filter((id) => positions.has(id));
  const anchors = anchorIds.map((id) => ({
    id,
    position: positions.get(id),
    size: sizes.get(id) || [0.8, 0.8, 1],
  }));

  if (!anchors.length) {
    const camera = scene.camera || {};
    return {
      ...DEFAULT_VIEW,
      target: finiteVector(camera.target) || DEFAULT_VIEW.target.slice(),
      distance: clamp(finiteNumber(camera.distance, DEFAULT_VIEW.distance), 6, 36),
      azimuth: wrapDegrees(finiteNumber(camera.azimuth, DEFAULT_VIEW.azimuth)),
      elevation: clamp(finiteNumber(camera.elevation, DEFAULT_VIEW.elevation), 35, 62),
      agent_id: agentId || null,
      focus_ids: [],
    };
  }

  const bounds = anchorBounds(anchors);
  const extent = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 2.8);
  const camera = scene.camera || {};
  const minDistance = Math.max(7.2, Math.min(9, finiteNumber(camera.min_distance, 7.2)));
  const maxDistance = Math.max(minDistance, Math.min(36, finiteNumber(camera.max_distance, 36)));
  const target = [
    round((bounds.minX + bounds.maxX) * 0.5),
    round((bounds.minY + bounds.maxY) * 0.5),
    round(clamp((bounds.minZ + bounds.maxZ) * 0.5, 0.65, 2.4)),
  ];

  return {
    target,
    distance: round(clamp(extent * 1.32 + 2.2, minDistance, maxDistance)),
    azimuth: round(wrapDegrees(finiteNumber(camera.azimuth, DEFAULT_VIEW.azimuth))),
    elevation: extent > 10 ? 54 : extent > 6 ? 50 : 46,
    panX: 0,
    panY: 0,
    agent_id: agentId || null,
    focus_ids: focusIds,
  };
}

function collectFocusIds(evidence, positions) {
  const explicit = unique((evidence.focus_ids || []).map((value) => String(value || "")))
    .filter((id) => positions.has(id));
  if (explicit.length) return explicit.slice(0, 4);
  const eventIds = unique(
    (evidence.events || []).map((event) => String(event?.object_id || "")).filter(Boolean),
  ).filter((id) => positions.has(id));
  if (evidence.kind === "event" && eventIds.length) return eventIds.slice(0, 4);
  const values = [];
  for (const id of evidence.objective_focus?.subject_ids || []) values.push(id);
  for (const id of evidence.objective_focus?.target_ids || []) values.push(id);
  if (evidence.interaction_guidance?.subject_id) values.push(evidence.interaction_guidance.subject_id);
  if (evidence.interaction_guidance?.target_id) values.push(evidence.interaction_guidance.target_id);
  if (evidence.navigation?.primary_target_id) values.push(evidence.navigation.primary_target_id);
  values.push(...eventIds);
  return unique(values.map((value) => String(value || "")).filter(Boolean))
    .filter((id) => positions.has(id))
    .slice(0, 4);
}

function expandMechanismFocusIds(focusIds, evidence, mechanisms, positions) {
  const values = [...focusIds];
  const mechanismId = String(evidence.objective_focus?.current_metrics?.mechanism_id || "");
  for (const mechanism of mechanisms || []) {
    const id = String(mechanism?.id || "");
    const triggerId = String(mechanism?.trigger_id || "");
    const gateId = String(mechanism?.gate_id || "");
    const related = (mechanismId && mechanismId === id)
      || values.includes(id)
      || values.includes(triggerId)
      || values.includes(gateId);
    if (!related) continue;
    if (triggerId && positions.has(triggerId)) values.push(triggerId);
    if (gateId && positions.has(gateId)) values.push(gateId);
  }
  return unique(values).filter((id) => positions.has(id)).slice(0, 4);
}

function completedObjectiveCheck(frame) {
  const checks = Array.isArray(frame.objective?.checks) ? frame.objective.checks : [];
  const label = String(frame.label || "").replace(/^Completed:\s*/i, "").trim().toLowerCase();
  if (label) {
    const exact = checks.find((check) => (
      check?.passed
      && String(check.description || check.id || "").trim().toLowerCase() === label
    ));
    if (exact) return exact;
  }
  const passed = checks.filter((check) => check?.passed);
  return passed.at(-1) || null;
}

function anchorBounds(anchors) {
  const bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
    minZ: Infinity,
    maxZ: -Infinity,
  };
  for (const anchor of anchors) {
    const [x, y, z] = anchor.position;
    const [width, depth, height] = anchor.size;
    const padding = anchor.id === anchors[0]?.id ? 0.75 : 1;
    bounds.minX = Math.min(bounds.minX, x - width * 0.5 - padding);
    bounds.maxX = Math.max(bounds.maxX, x + width * 0.5 + padding);
    bounds.minY = Math.min(bounds.minY, y - depth * 0.5 - padding);
    bounds.maxY = Math.max(bounds.maxY, y + depth * 0.5 + padding);
    bounds.minZ = Math.min(bounds.minZ, z - height * 0.5);
    bounds.maxZ = Math.max(bounds.maxZ, z + height * 0.5);
  }
  return bounds;
}

function finiteVector(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const vector = value.slice(0, 3).map(Number);
  return vector.every(Number.isFinite) ? vector : null;
}

function finiteSize(value) {
  const vector = finiteVector(value);
  return vector ? vector.map((item) => Math.max(0.02, Math.abs(item))) : [0.8, 0.8, 1];
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function wrapDegrees(value) {
  return ((Number(value || 0) + 180) % 360 + 360) % 360 - 180;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Number(value)));
}

function round(value) {
  return Math.round(Number(value) * 1000) / 1000;
}
