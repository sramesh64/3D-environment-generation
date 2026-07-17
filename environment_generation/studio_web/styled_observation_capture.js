import { VisualPreviewRenderer } from "./visual_renderer.js";

const stage = document.getElementById("captureStage");
const renderer = new VisualPreviewRenderer(stage, {
  captureMode: true,
  fixedSize: [640, 360],
  clampView: false,
});

window.environmentGenerationStyledObservation = {
  async setScene(visualScene) {
    await renderer.setScene(visualScene);
    return true;
  },

  render(payload) {
    renderer.applyPhysicsState(payload?.objects || []);
    renderer.applyGameState({ mechanisms: payload?.mechanisms || [] });
    for (const sourceId of payload?.hidden_source_ids || []) {
      renderer.setObjectVisibility(sourceId, false);
    }
    renderer.setCameraPose(payload?.camera || {});
    return renderer.capturePixelStats();
  },
};

window.dispatchEvent(new Event("environment-generation-styled-observation-ready"));
