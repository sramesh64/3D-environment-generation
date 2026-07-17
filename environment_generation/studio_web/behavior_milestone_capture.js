import { VisualPreviewRenderer } from "./visual_renderer.js";
import {
  buildBehaviorMilestoneView,
  inheritBehaviorMilestoneFocus,
  nearestTrajectoryFrame,
} from "./behavior_milestone_view.js";

export const BEHAVIOR_MILESTONE_IMAGE_SIZE = [480, 270];

const captureCache = new Map();

export function captureBehaviorMilestones({ visualScene, evidenceFrames, trajectoryUrl, mechanisms = [] }) {
  const frames = Array.isArray(evidenceFrames) ? evidenceFrames.filter(Boolean).slice(0, 6) : [];
  if (!visualScene || !trajectoryUrl || !frames.length) return Promise.resolve([]);
  const mechanismKey = (mechanisms || [])
    .map((item) => `${item?.id || ""}:${item?.trigger_id || ""}:${item?.gate_id || ""}`)
    .join(",");
  const cacheKey = `${trajectoryUrl}|${frames.map((frame) => `${Number(frame.attempt || 1)}:${Number(frame.step || 0)}`).join(",")}|${mechanismKey}`;
  if (captureCache.has(cacheKey)) return captureCache.get(cacheKey);
  const request = renderMilestones({ visualScene, evidenceFrames: frames, trajectoryUrl, mechanisms })
    .catch((error) => {
      captureCache.delete(cacheKey);
      throw error;
    });
  captureCache.set(cacheKey, request);
  while (captureCache.size > 10) captureCache.delete(captureCache.keys().next().value);
  return request;
}

async function renderMilestones({ visualScene, evidenceFrames, trajectoryUrl, mechanisms }) {
  const response = await fetch(trajectoryUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`Trajectory request failed: ${response.status}`);
  const trajectory = await response.json();
  const trajectoryFrames = Array.isArray(trajectory.frames) ? trajectory.frames : [];
  if (!trajectoryFrames.length) throw new Error("Trajectory contains no replay frames");

  const host = document.createElement("div");
  const renderer = new VisualPreviewRenderer(host, {
    captureMode: true,
    fixedSize: BEHAVIOR_MILESTONE_IMAGE_SIZE,
    clampView: false,
  });
  try {
    await renderer.setScene(visualScene);
    const captures = [];
    const focusedEvidenceFrames = inheritBehaviorMilestoneFocus(evidenceFrames);
    for (const evidenceFrame of focusedEvidenceFrames) {
      const trajectoryFrame = nearestTrajectoryFrame(
        trajectoryFrames,
        evidenceFrame.step,
        evidenceFrame.attempt,
      );
      if (!trajectoryFrame) continue;
      const objects = physicsObjectsWithEvidenceAgent(trajectoryFrame, evidenceFrame);
      const mechanismState = trajectoryFrame.mechanisms || [];
      renderer.applyPhysicsState(objects);
      renderer.applyGameState({ mechanisms: mechanismState });
      const view = buildBehaviorMilestoneView({
        visualScene,
        evidenceFrame,
        trajectoryFrame,
        mechanisms,
      });
      renderer.setView(view);
      const capture = await renderer.capturePng();
      captures.push({
        evidence_index: Number(evidenceFrame.index || captures.length),
        evidence_step: Number(evidenceFrame.step || 0),
        trajectory_step: Number(trajectoryFrame.total_step || 0),
        trajectory_attempt: Number(trajectoryFrame.attempt || evidenceFrame.attempt || 1),
        view,
        objects,
        mechanisms: mechanismState,
        image_data_url: await blobToDataUrl(capture.blob),
      });
    }
    return captures;
  } finally {
    renderer.dispose();
  }
}

function physicsObjectsWithEvidenceAgent(trajectoryFrame, evidenceFrame) {
  const agentId = String(evidenceFrame.agent?.id || "");
  const agentPosition = Array.isArray(evidenceFrame.agent?.position)
    ? evidenceFrame.agent.position.map(Number)
    : null;
  if (!agentId || !agentPosition?.every(Number.isFinite)) return trajectoryFrame.objects || [];
  return (trajectoryFrame.objects || []).map((object) => (
    String(object?.id || "") === agentId
      ? { ...object, position: agentPosition }
      : object
  ));
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
    reader.addEventListener("error", () => reject(reader.error || new Error("Could not read milestone image")), { once: true });
    reader.readAsDataURL(blob);
  });
}
