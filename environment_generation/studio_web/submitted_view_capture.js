import { VisualPreviewRenderer } from "./visual_renderer.js";

const MAX_CAPTURE_WIDTH = 960;
const MAX_CAPTURE_HEIGHT = 720;
const MIN_CAPTURE_EDGE = 240;

export async function captureSubmittedView({ visualScene, physicsObjects = [], view, viewport }) {
  if (!visualScene) throw new Error("The submitted view has no visual scene");
  const fixedSize = captureSize(viewport);
  const host = document.createElement("div");
  const renderer = new VisualPreviewRenderer(host, {
    captureMode: true,
    fixedSize,
    clampView: false,
  });
  try {
    await renderer.setScene(visualScene);
    renderer.setView(view);
    const screenSpace = renderer.describeScreenSpace(physicsObjects);
    const capture = await renderer.capturePng();
    return {
      image_data_url: await blobToDataUrl(capture.blob),
      image: {
        width: capture.stats.width,
        height: capture.stats.height,
        mime_type: "image/png",
        capture_kind: "exact_authored_before_view",
      },
      screen_space: screenSpace,
    };
  } finally {
    renderer.dispose();
  }
}

export function captureSize(viewport) {
  const width = Math.max(1, Number(viewport?.width || 16));
  const height = Math.max(1, Number(viewport?.height || 9));
  const aspect = width / height;
  let captureWidth;
  let captureHeight;
  if (aspect >= 1) {
    captureWidth = MAX_CAPTURE_WIDTH;
    captureHeight = Math.round(captureWidth / aspect);
    if (captureHeight > MAX_CAPTURE_HEIGHT) {
      captureHeight = MAX_CAPTURE_HEIGHT;
      captureWidth = Math.round(captureHeight * aspect);
    }
  } else {
    captureHeight = MAX_CAPTURE_HEIGHT;
    captureWidth = Math.round(captureHeight * aspect);
  }
  if (Math.min(captureWidth, captureHeight) < MIN_CAPTURE_EDGE) {
    const scale = MIN_CAPTURE_EDGE / Math.min(captureWidth, captureHeight);
    captureWidth = Math.round(captureWidth * scale);
    captureHeight = Math.round(captureHeight * scale);
  }
  return [captureWidth, captureHeight];
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
    reader.addEventListener("error", () => reject(reader.error || new Error("Could not read submitted view image")), { once: true });
    reader.readAsDataURL(blob);
  });
}
