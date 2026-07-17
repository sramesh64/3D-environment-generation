import { VisualPreviewRenderer } from "./visual_renderer.js";

export const VISUAL_REVIEW_IMAGE_SIZE = [960, 540];
export const VISUAL_REVIEW_VIEW_IDS = ["primary", "reverse", "layout"];

export async function captureVisualReviewEvidence({ pending, beforeScene = null, afterScene }) {
  if (!pending?.review_id || !afterScene) throw new Error("Visual review capture is missing scene context");
  const kind = pending.kind === "revision" ? "revision" : "initial";
  const paired = Boolean(pending.before);
  if (paired && !beforeScene) throw new Error("Paired visual review is missing the before scene");
  const views = buildReviewViewDescriptors({
    beforeScene,
    afterScene,
    submittedViewContext: pending.submitted_view_context,
    useSubmittedView: paired || kind === "revision",
  });

  const host = document.createElement("div");
  const renderer = new VisualPreviewRenderer(host, {
    captureMode: true,
    fixedSize: VISUAL_REVIEW_IMAGE_SIZE,
    clampView: false,
  });
  try {
    const capturedViews = [];
    for (const view of views) {
      let beforeImage = null;
      let beforeStats = null;
      if (paired) {
        await renderer.setScene(beforeScene);
        renderer.setView(view.camera);
        const capture = await renderer.capturePng();
        beforeImage = await blobToDataUrl(capture.blob);
        beforeStats = capture.stats;
      }
      await renderer.setScene(afterScene);
      renderer.setView(view.camera);
      const afterCapture = await renderer.capturePng();
      capturedViews.push({
        id: view.id,
        camera: view.camera,
        before_image: beforeImage,
        after_image: await blobToDataUrl(afterCapture.blob),
        capture_stats: {
          before: beforeStats,
          after: afterCapture.stats,
        },
      });
    }
    return {
      stream: true,
      scene_version: {
        spec_hash: pending.after?.spec_hash || "",
        visual_scene_hash: pending.after?.visual_scene_hash || "",
      },
      views: capturedViews,
    };
  } finally {
    renderer.dispose();
  }
}

export function buildReviewViewDescriptors({
  beforeScene = null,
  afterScene,
  submittedViewContext = null,
  kind = "initial",
  useSubmittedView = null,
}) {
  const defaultCamera = afterScene?.camera || {};
  const submittedVisual = submittedViewContext?.visual || {};
  const shouldUseSubmitted = useSubmittedView === null ? kind === "revision" : Boolean(useSubmittedView);
  const useSubmitted = shouldUseSubmitted && hasFiniteView(submittedVisual);
  const primary = normalizeView(useSubmitted ? {
    target: submittedVisual.target,
    distance: submittedVisual.distance,
    azimuth: submittedVisual.azimuth_degrees,
    elevation: submittedVisual.elevation_degrees,
    panX: submittedVisual.pan_x_world,
    panY: submittedVisual.pan_z_world,
  } : {
    target: defaultCamera.target,
    distance: defaultCamera.distance,
    azimuth: defaultCamera.azimuth,
    elevation: defaultCamera.elevation,
    panX: 0,
    panY: 0,
  });
  const bounds = combinedSceneBounds([beforeScene, afterScene].filter(Boolean));
  const fittedDistance = Math.max(9, bounds.extent * 1.08, bounds.height * 2.6);
  return [
    { id: "primary", camera: primary },
    {
      id: "reverse",
      camera: normalizeView({
        target: bounds.target,
        distance: fittedDistance,
        azimuth: wrapDegrees(primary.azimuth + 180),
        elevation: 38,
        panX: 0,
        panY: 0,
      }),
    },
    {
      id: "layout",
      camera: normalizeView({
        target: bounds.target,
        distance: fittedDistance * 1.08,
        azimuth: primary.azimuth,
        elevation: 68,
        panX: 0,
        panY: 0,
      }),
    },
  ];
}

function combinedSceneBounds(scenes) {
  const bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
    minZ: 0,
    maxZ: 1,
  };
  for (const scene of scenes) {
    const worldSize = Array.isArray(scene?.world_size) ? scene.world_size.map(Number) : [12, 12, 4];
    expandBounds(bounds, [0, 0, Number(worldSize[2] || 4) * 0.5], worldSize);
    for (const object of scene?.objects || []) {
      if (!Array.isArray(object?.position) || !Array.isArray(object?.size)) continue;
      expandBounds(bounds, object.position.map(Number), object.size.map(Number));
    }
  }
  if (!Number.isFinite(bounds.minX)) return { target: [0, 0, 0.5], extent: 12, height: 4 };
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const depth = Math.max(1, bounds.maxY - bounds.minY);
  const height = Math.max(1, bounds.maxZ - bounds.minZ);
  return {
    target: [
      roundNumber((bounds.minX + bounds.maxX) * 0.5),
      roundNumber((bounds.minY + bounds.maxY) * 0.5),
      roundNumber(Math.max(0.5, bounds.minZ + height * 0.32)),
    ],
    extent: Math.max(width, depth),
    height,
  };
}

function expandBounds(bounds, position, size) {
  const halfX = Math.abs(Number(size[0] || 0)) * 0.5;
  const halfY = Math.abs(Number(size[1] || 0)) * 0.5;
  const halfZ = Math.abs(Number(size[2] || 0)) * 0.5;
  bounds.minX = Math.min(bounds.minX, Number(position[0] || 0) - halfX);
  bounds.maxX = Math.max(bounds.maxX, Number(position[0] || 0) + halfX);
  bounds.minY = Math.min(bounds.minY, Number(position[1] || 0) - halfY);
  bounds.maxY = Math.max(bounds.maxY, Number(position[1] || 0) + halfY);
  bounds.minZ = Math.min(bounds.minZ, Number(position[2] || 0) - halfZ);
  bounds.maxZ = Math.max(bounds.maxZ, Number(position[2] || 0) + halfZ);
}

function hasFiniteView(value) {
  return Array.isArray(value?.target)
    && value.target.length === 3
    && [value.distance, value.azimuth_degrees, value.elevation_degrees].every((item) => Number.isFinite(Number(item)));
}

function normalizeView(value) {
  const target = Array.isArray(value?.target) && value.target.length === 3 ? value.target : [0, 0, 0.5];
  return {
    target: target.map((item) => roundNumber(Number(item || 0))),
    distance: roundNumber(Math.max(1, Number(value?.distance || 14))),
    azimuth: roundNumber(wrapDegrees(Number(value?.azimuth || 0))),
    elevation: roundNumber(Math.max(5, Math.min(89, Number(value?.elevation || 38)))),
    panX: roundNumber(Number(value?.panX || 0)),
    panY: roundNumber(Number(value?.panY || 0)),
  };
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
    reader.addEventListener("error", () => reject(reader.error || new Error("Could not read visual review image")), { once: true });
    reader.readAsDataURL(blob);
  });
}

function wrapDegrees(value) {
  return ((Number(value || 0) + 180) % 360 + 360) % 360 - 180;
}

function roundNumber(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}
