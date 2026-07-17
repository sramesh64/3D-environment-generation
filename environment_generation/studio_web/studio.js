import { VisualPreviewRenderer } from "./visual_renderer.js";
import { captureVisualReviewEvidence } from "./visual_review_capture.js";
import { captureSubmittedView } from "./submitted_view_capture.js";
import { PRIMITIVE_CATALOG, primitivePreviewScene } from "./primitive_catalog.js";
import { captureBehaviorMilestones } from "./behavior_milestone_capture.js";
import {
  behaviorReplayFrameState,
  nearestTrajectoryFrameIndex,
} from "./behavior_milestone_view.js";
import {
  buildBehaviorHeaderView,
  buildBehaviorOutcomeView,
  buildBehaviorRunView,
  describeBehaviorCheck,
  describeBehaviorMilestone,
  formatBehaviorMetric,
} from "./behavior_trial_view.js";
import {
  recordingFinishIntent,
  recordingFinishView,
  recordingFrameCount,
  recordingReportRequested,
} from "./task_recording.js";
import { buildTaskResultView } from "./task_view.js";
import {
  normalizeTaskActivityTimeline,
  taskActivityAtStep,
  taskTrajectoryRuns,
} from "./task_trajectory_replay.js";

const $ = (id) => document.getElementById(id);
const MODEL_STORAGE_KEY = "environment_generation.codex_model";
const behaviorReplaySelections = new WeakMap();

const state = {
  scenes: [],
  currentScene: null,
  route: { page: "home" },
  codexModels: {
    items: [],
    defaultModel: null,
    selected: readStoredCodexModel(),
    status: "loading",
    message: "",
  },
  preview: {
    mode: "visual",
    sceneId: "",
    sceneVersion: "",
    renderer: null,
    ready: Promise.resolve(false),
    visualScene: null,
    visual: {
      azimuth: -42,
      elevation: 38,
      distance: 14,
      defaultDistance: 14,
      panX: 0,
      panY: 0,
    },
    url: "",
    zoom: 1,
    x: 0,
    y: 0,
    orbitFrames: [],
    orbitIndex: 0,
    dragging: false,
    dragMode: "",
    pointerId: null,
    lastX: 0,
    lastY: 0,
  },
  createPreview: {
    renderer: null,
    ready: null,
    visualScene: null,
    spec: null,
    view: null,
    dragging: false,
    dragMode: "",
    pointerId: null,
    lastX: 0,
    lastY: 0,
  },
  primitiveBrowser: {
    open: false,
    returnFocus: null,
    selectedId: "pushable_box",
    renderer: null,
    scene: null,
    view: null,
    dragging: false,
    pointerId: null,
    lastX: 0,
    lastY: 0,
  },
  inspectorTab: "checks",
  running: false,
  generating: false,
  revising: false,
  launchingPlay: false,
  visualReview: {
    activeIds: new Set(),
    activityByEnv: new Map(),
    pollTimers: new Map(),
  },
  behavior: {
    requests: new Map(),
    activity: [],
    pollTimer: null,
    replay: {
      active: false,
      frames: [],
      index: 0,
      playing: false,
      timer: null,
      trialId: "",
      loadToken: 0,
      previousView: null,
      kind: "behavior",
      taskActivity: [],
      taskId: "",
      taskInstruction: "",
      runId: "",
      runNumber: 0,
      overlayState: null,
    },
  },
  tasks: {
    composerOpen: false,
    compiling: false,
    runningTaskId: "",
    runModels: new Map(),
    expandedRuns: new Set(),
    selectedResults: new Map(),
    resultReports: new Map(),
    activeRun: null,
    notice: null,
    oracle: {
      active: false,
      sessionId: "",
      taskId: "",
      instruction: "",
      agentId: "",
      keys: new Set(),
      keyPulseUntil: new Map(),
      timer: null,
      requestInFlight: false,
      terminalReason: "",
      report: null,
      hasStarted: false,
      hasActions: false,
      idleTicks: 0,
      readyToValidate: false,
      reportCurrent: true,
      lastTickAt: 0,
      simulationTimestep: 0.01,
      finishPending: false,
      finishing: false,
    },
  },
  play: {
    active: false,
    sessionId: "",
    envId: "",
    agentId: "",
    status: "",
    keys: new Set(),
    keyPulseUntil: new Map(),
    timer: null,
    requestInFlight: false,
    gameState: "",
    resetTimer: null,
    eventTimer: null,
  },
};

const MIN_PREVIEW_ZOOM = 1;
const MAX_PREVIEW_ZOOM = 5;
const PREVIEW_ZOOM_STEP = 0.25;
const PLAY_TICK_MS = 40;
const PLAY_CAMERA_DISTANCE = 7.2;
const PLAY_CAMERA_ELEVATION = 28;
const ORACLE_IDLE_TICKS_AFTER_INPUT = 30;

const els = {
  homePage: $("homePage"),
  createPage: $("createPage"),
  createPreviewStage: $("createPreviewStage"),
  createPrimitiveBrowserButton: $("createPrimitiveBrowserButton"),
  primitiveBrowserButton: $("primitiveBrowserButton"),
  primitiveBrowser: $("primitiveBrowser"),
  primitiveBrowserBackdrop: $("primitiveBrowserBackdrop"),
  primitiveBrowserClose: $("primitiveBrowserClose"),
  primitiveCatalogList: $("primitiveCatalogList"),
  primitivePreviewStage: $("primitivePreviewStage"),
  primitivePreviewReset: $("primitivePreviewReset"),
  primitiveCategory: $("primitiveCategory"),
  primitiveName: $("primitiveName"),
  primitiveTraits: $("primitiveTraits"),
  primitiveDescription: $("primitiveDescription"),
  primitiveUse: $("primitiveUse"),
  envPage: $("envPage"),
  homeLink: $("homeLink"),
  newEnvButton: $("newEnvButton"),
  homeCreateButton: $("homeCreateButton"),
  emptyCreateButton: $("emptyCreateButton"),
  cancelCreateButton: $("cancelCreateButton"),
  backHomeButton: $("backHomeButton"),
  playEnvButton: $("playEnvButton"),
  variationButton: $("variationButton"),
  levelMeta: $("levelMeta"),
  sceneCount: $("sceneCount"),
  sceneGrid: $("sceneGrid"),
  emptyLibrary: $("emptyLibrary"),
  modelPicker: $("modelPicker"),
  modelSelect: $("modelSelect"),
  topStatus: $("topStatus"),
  activeTitle: $("activeTitle"),
  sceneLabel: $("sceneLabel"),
  previewStage: $("previewStage"),
  movementHelp: $("movementHelp"),
  movementHelpMode: $("movementHelpMode"),
  movementHelpExit: $("movementHelpExit"),
  behaviorReplayControls: $("behaviorReplayControls"),
  behaviorReplayToggle: $("behaviorReplayToggle"),
  behaviorReplaySlider: $("behaviorReplaySlider"),
  behaviorReplayLabel: $("behaviorReplayLabel"),
  behaviorReplayExit: $("behaviorReplayExit"),
  objectSummary: $("objectSummary"),
  objectList: $("objectList"),
  revisionMessages: $("revisionMessages"),
  conversationTitle: $("conversationTitle"),
  conversationSubtitle: $("conversationSubtitle"),
  revisionForm: $("revisionForm"),
  revisionPrompt: $("revisionPrompt"),
  revisionButton: $("revisionButton"),
  generateForm: $("generateForm"),
  generateButton: $("generateButton"),
  envName: $("envName"),
  envNameError: $("envNameError"),
  prompt: $("prompt"),
  gameOverlay: $("gameOverlay"),
  gameOverlayTitle: $("gameOverlayTitle"),
  gameOverlayDetail: $("gameOverlayDetail"),
  gameOverlayAction: $("gameOverlayAction"),
  playEventNotice: $("playEventNotice"),
  taskRunOverlay: $("taskRunOverlay"),
  taskRunOverlayTitle: $("taskRunOverlayTitle"),
  taskRunOverlayStatus: $("taskRunOverlayStatus"),
  taskRunOverlayLog: $("taskRunOverlayLog"),
  taskRunOverlayDetail: $("taskRunOverlayDetail"),
  envActivityState: $("envActivityState"),
  envActivityLog: $("envActivityLog"),
  envCheckSummary: $("envCheckSummary"),
  envCheckStats: $("envCheckStats"),
  envCheckList: $("envCheckList"),
  visualReviewSummary: $("visualReviewSummary"),
  visualReviewStats: $("visualReviewStats"),
  visualReviewBody: $("visualReviewBody"),
  visualReviewRepair: $("visualReviewRepair"),
  visualReviewRetry: $("visualReviewRetry"),
  behaviorTrialsSummary: $("behaviorTrialsSummary"),
  behaviorTrialStats: $("behaviorTrialStats"),
  behaviorTrialsBody: $("behaviorTrialsBody"),
  behaviorRunAll: $("behaviorRunAll"),
  behaviorDismissAll: $("behaviorDismissAll"),
  behaviorRepair: $("behaviorRepair"),
  taskSummary: $("taskSummary"),
  taskTabBadge: $("taskTabBadge"),
  newTaskButton: $("newTaskButton"),
  taskCreateForm: $("taskCreateForm"),
  taskInstruction: $("taskInstruction"),
  taskCreateCancel: $("taskCreateCancel"),
  taskCreateSubmit: $("taskCreateSubmit"),
  taskNotice: $("taskNotice"),
  taskList: $("taskList"),
  taskOracleControls: $("taskOracleControls"),
  taskOracleInstruction: $("taskOracleInstruction"),
  taskOracleChecks: $("taskOracleChecks"),
  taskOracleChecklist: $("taskOracleChecklist"),
  taskOracleReset: $("taskOracleReset"),
  taskOracleCancel: $("taskOracleCancel"),
  taskOracleFinish: $("taskOracleFinish"),
  checkTabBadge: $("checkTabBadge"),
  visualTabBadge: $("visualTabBadge"),
  behaviorTabBadge: $("behaviorTabBadge"),
  objectTabBadge: $("objectTabBadge"),
  activityTabBadge: $("activityTabBadge"),
  inspectorTooltip: $("inspectorTooltip"),
  agentState: $("agentState"),
};

els.homeLink.addEventListener("click", navigateHome);
els.modelSelect.addEventListener("change", () => {
  state.codexModels.selected = String(els.modelSelect.value || "");
  persistCodexModel(state.codexModels.selected);
  renderModelSelector();
  if (state.route.page === "env" && state.currentScene) renderTasks(state.currentScene);
});
els.primitiveBrowserButton.addEventListener("click", (event) => void openPrimitiveBrowser(event.currentTarget));
els.createPrimitiveBrowserButton.addEventListener("click", (event) => void openPrimitiveBrowser(event.currentTarget));
els.primitiveBrowserClose.addEventListener("click", closePrimitiveBrowser);
els.primitiveBrowserBackdrop.addEventListener("click", closePrimitiveBrowser);
els.primitiveCatalogList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-primitive-id]");
  if (!button) return;
  void selectPrimitive(button.dataset.primitiveId || "");
});
els.primitivePreviewReset.addEventListener("click", resetPrimitivePreview);
els.primitivePreviewStage.addEventListener("wheel", handlePrimitivePreviewWheel, { passive: false });
els.primitivePreviewStage.addEventListener("pointerdown", startPrimitivePreviewDrag);
els.primitivePreviewStage.addEventListener("pointermove", movePrimitivePreviewDrag);
els.primitivePreviewStage.addEventListener("pointerup", stopPrimitivePreviewDrag);
els.primitivePreviewStage.addEventListener("pointercancel", stopPrimitivePreviewDrag);
els.primitivePreviewStage.addEventListener("lostpointercapture", stopPrimitivePreviewDrag);
els.newEnvButton.addEventListener("click", navigateCreate);
els.homeCreateButton.addEventListener("click", navigateCreate);
els.emptyCreateButton.addEventListener("click", navigateCreate);
els.cancelCreateButton.addEventListener("click", navigateHome);
els.backHomeButton.addEventListener("click", navigateHome);
els.playEnvButton.addEventListener("click", togglePlayableEnvironment);
els.variationButton.addEventListener("click", () => void createVariation());
els.gameOverlayAction.addEventListener("click", () => void resetPlayableEnvironment());
els.visualReviewRetry.addEventListener("click", () => {
  void retryCurrentVisualReview();
});
els.visualReviewRepair.addEventListener("click", () => void repairFromVisualReview());
els.behaviorRunAll.addEventListener("click", () => void runAllBehaviorTrials());
els.behaviorDismissAll.addEventListener("click", () => void dismissBehaviorTrials());
els.behaviorRepair.addEventListener("click", () => void repairFromBehaviorTrials());
els.behaviorTrialsBody.addEventListener("click", handleBehaviorTrialAction);
els.newTaskButton.addEventListener("click", openTaskComposer);
els.taskCreateCancel.addEventListener("click", closeTaskComposer);
els.taskCreateForm.addEventListener("submit", (event) => void submitTaskDefinition(event));
els.taskList.addEventListener("click", (event) => void handleTaskAction(event));
els.taskList.addEventListener("change", handleTaskRunModelChange);
document.addEventListener("click", closeOpenTaskActionMenus);
els.taskOracleReset.addEventListener("click", () => void resetTaskOracle());
els.taskOracleCancel.addEventListener("click", () => void cancelTaskOracle());
els.taskOracleFinish.addEventListener("click", () => void finishTaskOracle());
els.behaviorReplayToggle.addEventListener("click", toggleBehaviorReplay);
els.behaviorReplaySlider.addEventListener("input", () => setBehaviorReplayFrame(Number(els.behaviorReplaySlider.value)));
els.behaviorReplayExit.addEventListener("click", exitBehaviorReplay);

for (const button of document.querySelectorAll("[data-inspector-tab]")) {
  button.addEventListener("click", () => setInspectorTab(button.dataset.inspectorTab || "checks"));
  const info = button.querySelector(".tab-info");
  if (info) {
    info.addEventListener("mouseenter", () => showInspectorTooltip(info));
    info.addEventListener("mouseleave", hideInspectorTooltip);
  }
}

window.addEventListener("scroll", hideInspectorTooltip, true);
window.addEventListener("resize", hideInspectorTooltip);

els.previewStage.addEventListener("wheel", handlePreviewWheel, { passive: false });
els.previewStage.addEventListener("pointerdown", startPreviewDrag);
els.previewStage.addEventListener("pointermove", movePreviewDrag);
els.previewStage.addEventListener("pointerup", stopPreviewDrag);
els.previewStage.addEventListener("pointercancel", stopPreviewDrag);
els.previewStage.addEventListener("lostpointercapture", stopPreviewDrag);
els.createPreviewStage.addEventListener("wheel", handleCreatePreviewWheel, { passive: false });
els.createPreviewStage.addEventListener("pointerdown", startCreatePreviewDrag);
els.createPreviewStage.addEventListener("pointermove", moveCreatePreviewDrag);
els.createPreviewStage.addEventListener("pointerup", stopCreatePreviewDrag);
els.createPreviewStage.addEventListener("pointercancel", stopCreatePreviewDrag);
els.createPreviewStage.addEventListener("lostpointercapture", stopCreatePreviewDrag);
window.addEventListener("keydown", handlePrimitiveBrowserKeyDown);
window.addEventListener("keydown", handlePlayKeyDown);
window.addEventListener("keyup", handlePlayKeyUp);
window.addEventListener("blur", () => {
  state.play.keys.clear();
  state.tasks.oracle.keys.clear();
});
window.addEventListener("pagehide", stopPlayOnPageExit);
window.addEventListener("pagehide", stopTaskOracleOnPageExit);

renderPrimitiveCatalogList();

els.generateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.running) return;
  const form = new FormData(els.generateForm);
  const payload = {
    name: String(form.get("name") || "").trim(),
    prompt: String(form.get("prompt") || ""),
  };
  if (!payload.name) {
    setEnvNameValidation("Environment must have a name.");
    els.envName.focus();
    return;
  }
  setEnvNameValidation();
  if (!payload.prompt.trim()) return;
  await generateEnvironment(payload);
});
els.envName.addEventListener("input", () => {
  if (els.envName.value.trim()) setEnvNameValidation();
});

els.revisionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.revisionPrompt.value.trim();
  if (!message || state.running || !state.currentScene?.env_id) return;
  els.revisionPrompt.value = "";
  resizeRevisionPrompt();
  await reviseEnvironment(message);
});
els.revisionPrompt.addEventListener("input", resizeRevisionPrompt);

window.addEventListener("hashchange", () => {
  const nextRoute = parseRoute();
  if (state.behavior.pollTimer !== null) {
    window.clearTimeout(state.behavior.pollTimer);
    state.behavior.pollTimer = null;
  }
  if (state.behavior.replay.active && (nextRoute.page !== "env" || nextRoute.envId !== state.currentScene?.env_id)) {
    exitBehaviorReplay();
  }
  if (state.play.active && (nextRoute.page !== "env" || nextRoute.envId !== state.play.envId)) {
    void stopPlayableEnvironment({ silent: true, resetPreview: false });
  }
  if (state.tasks.oracle.active && (nextRoute.page !== "env" || nextRoute.envId !== state.currentScene?.env_id)) {
    void cancelTaskOracle({ silent: true, resetPreview: false });
  }
  state.route = nextRoute;
  render();
});

if (!window.location.hash) {
  history.replaceState(null, "", "#/");
}
state.route = parseRoute();
void loadCodexModels();
loadScenes().catch((error) => {
  setStatus("Failed", "error");
  setActivity("Failed", [{ type: "error", message: error.message || String(error) }]);
});

async function loadScenes(selectEnvId = "") {
  const response = await fetch("/api/scenes");
  const data = await response.json();
  state.scenes = data.scenes || [];
  if (selectEnvId) {
    state.currentScene = state.scenes.find((scene) => scene.env_id === selectEnvId) || state.currentScene;
  } else if (state.route.page === "env") {
    state.currentScene = state.scenes.find((scene) => scene.env_id === state.route.envId) || null;
  }
  render();
}

async function loadCodexModels() {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Model request failed: ${response.status}`);
    state.codexModels.items = Array.isArray(data.models) ? data.models : [];
    state.codexModels.defaultModel = data.default_model?.id ? data.default_model : null;
    state.codexModels.status = data.status === "ready" ? "ready" : "unavailable";
    state.codexModels.message = String(data.message || "");
    if (
      state.codexModels.status === "ready"
      && state.codexModels.selected
      && !state.codexModels.items.some((model) => model.id === state.codexModels.selected)
    ) {
      state.codexModels.selected = "";
      persistCodexModel("");
    }
  } catch (error) {
    state.codexModels.status = "unavailable";
    state.codexModels.message = error.message || String(error);
  }
  renderModelSelector();
  if (state.route.page === "env" && state.currentScene) renderTasks(state.currentScene);
}

function renderModelSelector() {
  const catalog = state.codexModels;
  const selected = catalog.selected;
  const defaultModelName = populateCodexModelOptions(els.modelSelect, selected, "Codex default");
  els.modelSelect.disabled = catalog.status === "loading";
  els.modelPicker.classList.toggle("is-loading", catalog.status === "loading");
  els.modelPicker.classList.toggle("is-unavailable", catalog.status === "unavailable");
  const activeModel = catalog.items.find((model) => model.id === selected);
  const description = activeModel?.description
    || (selected
      ? `Use ${selected} for new Codex runs.`
      : defaultModelName
        ? `Use ${defaultModelName}, the default from your Codex config.`
        : "Use the default model selected by Codex.");
  const availability = catalog.status === "unavailable" && catalog.message ? ` ${catalog.message}` : "";
  els.modelPicker.title = `${description} Changes apply to new runs.${availability}`;
}

function populateCodexModelOptions(select, selected, defaultLabel = "Codex default") {
  const catalog = state.codexModels;
  select.replaceChildren();
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  const defaultModelName = String(catalog.defaultModel?.name || catalog.defaultModel?.id || "");
  defaultOption.textContent = defaultModelName ? `${defaultLabel} (${defaultModelName})` : defaultLabel;
  defaultOption.title = defaultModelName
    ? `Use ${defaultModelName}, the default from your Codex config.`
    : "Use the default model selected by Codex.";
  select.appendChild(defaultOption);

  for (const model of catalog.items) {
    if (!model?.id) continue;
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name || model.id;
    option.title = model.description || model.id;
    select.appendChild(option);
  }
  if (catalog.status === "unavailable" && selected) {
    const savedOption = document.createElement("option");
    savedOption.value = selected;
    savedOption.textContent = `${selected} (saved)`;
    select.appendChild(savedOption);
  }

  select.value = selected;
  return defaultModelName;
}

function selectedCodexModel() {
  return String(state.codexModels.selected || state.codexModels.defaultModel?.id || "");
}

function withCodexModel(payload = {}) {
  return { ...payload, model: selectedCodexModel() };
}

function readStoredCodexModel() {
  try {
    return String(window.localStorage.getItem(MODEL_STORAGE_KEY) || "");
  } catch {
    return "";
  }
}

function persistCodexModel(model) {
  try {
    if (model) window.localStorage.setItem(MODEL_STORAGE_KEY, model);
    else window.localStorage.removeItem(MODEL_STORAGE_KEY);
  } catch {
    // The selection still applies for this session when persistent storage is unavailable.
  }
}

function renderPrimitiveCatalogList() {
  els.primitiveCatalogList.replaceChildren();
  let activeCategory = "";
  for (const item of PRIMITIVE_CATALOG) {
    if (item.category !== activeCategory) {
      activeCategory = item.category;
      appendText(els.primitiveCatalogList, "div", "primitive-list-heading", activeCategory);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `primitive-list-item ${item.id === state.primitiveBrowser.selectedId ? "active" : ""}`.trim();
    button.dataset.primitiveId = item.id;
    button.setAttribute("aria-pressed", String(item.id === state.primitiveBrowser.selectedId));
    const swatch = document.createElement("span");
    swatch.className = "primitive-swatch";
    swatch.style.setProperty("--primitive-color", item.color);
    const label = document.createElement("span");
    label.className = "primitive-list-label";
    label.textContent = item.label;
    const physics = document.createElement("span");
    physics.className = "primitive-list-physics";
    physics.textContent = item.physicsLabel;
    button.append(swatch, label, physics);
    els.primitiveCatalogList.appendChild(button);
  }
}

async function openPrimitiveBrowser(trigger = els.primitiveBrowserButton) {
  if (state.primitiveBrowser.open) return;
  state.primitiveBrowser.open = true;
  state.primitiveBrowser.returnFocus = trigger;
  state.play.keys.clear();
  state.play.keyPulseUntil.clear();
  els.primitiveBrowser.classList.remove("hidden");
  setPrimitiveBrowserExpanded(true);
  document.body.classList.add("primitive-browser-open");
  renderPrimitiveCatalogList();
  await new Promise((resolve) => window.requestAnimationFrame(resolve));
  if (!state.primitiveBrowser.open) return;
  const renderer = new VisualPreviewRenderer(els.primitivePreviewStage);
  state.primitiveBrowser.renderer = renderer;
  await selectPrimitive(state.primitiveBrowser.selectedId);
  if (state.primitiveBrowser.open) els.primitiveBrowserClose.focus();
}

function closePrimitiveBrowser() {
  if (!state.primitiveBrowser.open) return;
  const returnFocus = state.primitiveBrowser.returnFocus;
  state.primitiveBrowser.open = false;
  state.primitiveBrowser.returnFocus = null;
  stopPrimitivePreviewDrag({});
  state.primitiveBrowser.renderer?.dispose();
  state.primitiveBrowser.renderer = null;
  state.primitiveBrowser.scene = null;
  state.primitiveBrowser.view = null;
  els.primitiveBrowser.classList.add("hidden");
  setPrimitiveBrowserExpanded(false);
  document.body.classList.remove("primitive-browser-open");
  const focusTarget = returnFocus?.isConnected && !returnFocus.closest(".hidden")
    ? returnFocus
    : els.primitiveBrowserButton;
  focusTarget?.focus();
}

function setPrimitiveBrowserExpanded(expanded) {
  for (const button of [els.primitiveBrowserButton, els.createPrimitiveBrowserButton]) {
    button?.setAttribute("aria-expanded", String(expanded));
  }
}

async function selectPrimitive(primitiveId) {
  const item = PRIMITIVE_CATALOG.find((candidate) => candidate.id === primitiveId);
  if (!item) return;
  state.primitiveBrowser.selectedId = item.id;
  for (const button of els.primitiveCatalogList.querySelectorAll("[data-primitive-id]")) {
    const active = button.dataset.primitiveId === item.id;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  els.primitiveCategory.textContent = item.category;
  els.primitiveName.textContent = item.label;
  els.primitiveDescription.textContent = item.description;
  els.primitiveUse.textContent = item.use;
  els.primitiveTraits.replaceChildren();
  appendText(els.primitiveTraits, "span", "primitive-trait", item.physicsLabel);
  appendText(els.primitiveTraits, "span", "primitive-trait", item.shapeLabel);
  els.primitivePreviewStage.dataset.primitiveId = item.id;

  const renderer = state.primitiveBrowser.renderer;
  if (!renderer) return;
  const scene = primitivePreviewScene(item);
  state.primitiveBrowser.scene = scene;
  resetPrimitivePreviewView(scene);
  await renderer.setScene(scene);
  if (!state.primitiveBrowser.open || state.primitiveBrowser.renderer !== renderer) return;
  applyPrimitivePreviewView();
}

function resetPrimitivePreview() {
  if (!state.primitiveBrowser.scene) return;
  resetPrimitivePreviewView(state.primitiveBrowser.scene);
  applyPrimitivePreviewView();
}

function resetPrimitivePreviewView(scene) {
  const camera = scene?.camera || {};
  state.primitiveBrowser.view = {
    target: camera.target || [0, 0, 0.5],
    distance: Number(camera.distance || 6.2),
    azimuth: Number(camera.azimuth ?? -38),
    elevation: Number(camera.elevation ?? 28),
    panX: 0,
    panY: 0,
  };
}

function applyPrimitivePreviewView() {
  const view = state.primitiveBrowser.view;
  if (!view) return;
  view.distance = clamp(view.distance, 3, 12);
  view.elevation = clamp(view.elevation, 18, 72);
  state.primitiveBrowser.renderer?.setView(view);
  els.primitivePreviewStage.dataset.cameraAzimuth = String(roundNumber(wrapDegrees(view.azimuth)));
  els.primitivePreviewStage.dataset.cameraElevation = String(roundNumber(view.elevation));
  els.primitivePreviewStage.dataset.cameraDistance = String(roundNumber(view.distance));
}

function handlePrimitivePreviewWheel(event) {
  if (!state.primitiveBrowser.open || !state.primitiveBrowser.view) return;
  event.preventDefault();
  state.primitiveBrowser.view.distance *= event.deltaY > 0 ? 1.1 : 0.9;
  applyPrimitivePreviewView();
}

function startPrimitivePreviewDrag(event) {
  if (!state.primitiveBrowser.open || !state.primitiveBrowser.view) return;
  state.primitiveBrowser.dragging = true;
  state.primitiveBrowser.pointerId = event.pointerId;
  state.primitiveBrowser.lastX = event.clientX;
  state.primitiveBrowser.lastY = event.clientY;
  els.primitivePreviewStage.classList.add("is-panning");
  els.primitivePreviewStage.setPointerCapture(event.pointerId);
}

function movePrimitivePreviewDrag(event) {
  if (!state.primitiveBrowser.dragging || state.primitiveBrowser.pointerId !== event.pointerId) return;
  const dx = event.clientX - state.primitiveBrowser.lastX;
  const dy = event.clientY - state.primitiveBrowser.lastY;
  state.primitiveBrowser.lastX = event.clientX;
  state.primitiveBrowser.lastY = event.clientY;
  state.primitiveBrowser.view.azimuth += dx * 0.35;
  state.primitiveBrowser.view.elevation += dy * 0.12;
  applyPrimitivePreviewView();
}

function stopPrimitivePreviewDrag(event) {
  if (event.pointerId !== undefined && state.primitiveBrowser.pointerId !== event.pointerId) return;
  state.primitiveBrowser.dragging = false;
  state.primitiveBrowser.pointerId = null;
  els.primitivePreviewStage.classList.remove("is-panning");
}

function handlePrimitiveBrowserKeyDown(event) {
  if (!state.primitiveBrowser.open || event.code !== "Escape") return;
  event.preventDefault();
  event.stopImmediatePropagation();
  closePrimitiveBrowser();
}

async function fetchScene(envId) {
  const response = await fetch(`/api/scenes/${encodeURIComponent(envId)}?v=${Date.now()}`, {
    cache: "no-store",
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `Request failed: ${response.status}`);
  if (!data.scene) throw new Error("Scene response did not include a scene");
  return data.scene;
}

async function togglePlayableEnvironment() {
  if (state.play.active) {
    await stopPlayableEnvironment();
    return;
  }
  await launchPlayableEnvironment();
}

async function launchPlayableEnvironment() {
  const scene = state.currentScene;
  if (!scene?.env_id || state.running || state.launchingPlay || scene.capabilities?.playable === false) return;

  state.launchingPlay = true;
  updatePlayButton(scene);
  setStatus("Starting Play", "running");
  try {
    if (state.preview.mode !== "visual") {
      state.preview.mode = "visual";
      renderPreview();
    }
    const response = await fetch("/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ env_id: scene.env_id }),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Request failed: ${response.status}`);
    state.play.active = true;
    state.play.sessionId = data.session_id;
    state.play.envId = scene.env_id;
    state.play.agentId = data.state?.agent_id || "";
    state.play.status = data.state?.status || "Exploring";
    state.play.gameState = data.state?.game?.state || "playing";
    state.play.keys.clear();
    state.play.keyPulseUntil.clear();
    activateAgentCamera();
    applyPlayState(data.state);
    updatePlayModeControls();
    schedulePlayTick(0);
    setStatus("Playing", "ready");
  } catch (error) {
    appendRevisionMessage("assistant", `Play mode could not start: ${error.message || String(error)}`);
    setStatus("Play failed", "error");
  } finally {
    state.launchingPlay = false;
    updatePlayButton(state.currentScene);
  }
}

async function stopPlayableEnvironment({ silent = false, resetPreview = true } = {}) {
  const sessionId = state.play.sessionId;
  if (state.play.timer !== null) window.clearTimeout(state.play.timer);
  if (state.play.resetTimer !== null) window.clearTimeout(state.play.resetTimer);
  if (state.play.eventTimer !== null) window.clearTimeout(state.play.eventTimer);
  state.play.active = false;
  state.play.sessionId = "";
  state.play.envId = "";
  state.play.agentId = "";
  state.play.status = "";
  state.play.timer = null;
  state.play.requestInFlight = false;
  state.play.gameState = "";
  state.play.resetTimer = null;
  state.play.eventTimer = null;
  state.play.keys.clear();
  state.play.keyPulseUntil.clear();
  delete els.previewStage.dataset.playStatus;
  delete els.previewStage.dataset.agentPosition;
  delete els.previewStage.dataset.grounded;
  delete els.previewStage.dataset.cameraMode;
  delete els.previewStage.dataset.cameraAzimuth;
  delete els.previewStage.dataset.cameraElevation;
  els.gameOverlay.classList.add("hidden");
  els.playEventNotice.classList.add("hidden");
  updatePlayModeControls();
  updatePlayButton(state.currentScene);
  if (resetPreview && state.route.page === "env") {
    resetVisualViewFromScene(state.preview.visualScene);
    renderPreview();
  }
  if (!silent && state.currentScene) setStatus(sceneStatusLabel(state.currentScene), sceneStatusClass(state.currentScene));
  if (!sessionId) return;
  try {
    await fetch(`/api/play/${encodeURIComponent(sessionId)}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      keepalive: true,
    });
  } catch {
    // Sessions expire server-side; stopping is best-effort during navigation.
  }
}

function schedulePlayTick(delay = PLAY_TICK_MS) {
  if (!state.play.active) return;
  if (state.play.timer !== null) window.clearTimeout(state.play.timer);
  state.play.timer = window.setTimeout(runPlayTick, delay);
}

async function runPlayTick() {
  if (!state.play.active || !state.play.sessionId) return;
  if (state.play.requestInFlight) {
    schedulePlayTick();
    return;
  }
  const sessionId = state.play.sessionId;
  const input = currentPlayInput();
  state.play.requestInFlight = true;
  try {
    const response = await fetch(`/api/play/${encodeURIComponent(sessionId)}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...input,
        camera_azimuth: state.preview.visual.azimuth,
      }),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Request failed: ${response.status}`);
    if (state.play.active && state.play.sessionId === sessionId) applyPlayState(data.state);
  } catch (error) {
    if (state.play.active && state.play.sessionId === sessionId) {
      await stopPlayableEnvironment({ silent: true });
      appendRevisionMessage("assistant", `Play mode stopped: ${error.message || String(error)}`);
      setStatus("Play stopped", "error");
    }
  } finally {
    state.play.requestInFlight = false;
    if (state.play.gameState === "playing") schedulePlayTick();
  }
}

function applyPlayState(playState) {
  if (!playState || !state.preview.renderer) return;
  state.play.agentId = playState.agent_id || state.play.agentId;
  state.play.status = playState.status || "Exploring";
  const game = playState.game || { state: "playing", metrics: {} };
  state.play.gameState = game.state || "playing";
  state.preview.renderer.applyPhysicsState(playState.objects || []);
  state.preview.renderer.applyGameState?.(game);
  const agentTransform = (playState.objects || []).find((item) => item.id === state.play.agentId);
  els.previewStage.dataset.playStatus = state.play.status;
  els.previewStage.dataset.agentPosition = JSON.stringify(agentTransform?.position || []);
  els.previewStage.dataset.grounded = String(Boolean(playState.grounded));
  showPlayEvents(game.events || []);
  if (game.state === "failed") {
    if (state.play.eventTimer !== null) window.clearTimeout(state.play.eventTimer);
    state.play.eventTimer = null;
    els.playEventNotice.classList.add("hidden");
    const reason = String(game.failure_reason || "out_of_bounds").replaceAll("_", " ");
    setStatus("Out of bounds", "error");
    showGameOverlay("Try again", `${reason}. Resetting the courtyard...`, true, "Reset now");
    if (state.play.resetTimer === null) {
      state.play.resetTimer = window.setTimeout(() => {
        state.play.resetTimer = null;
        void resetPlayableEnvironment();
      }, 900);
    }
  } else {
    els.gameOverlay.classList.add("hidden");
    setStatus("Playing", "ready");
  }
}

function showPlayEvents(events) {
  const entered = [...events].reverse().find((event) => (
    event?.type === "zone_entered"
    && (!state.play.agentId || event.subject_id === state.play.agentId)
  ));
  if (!entered) return;
  if (!els.playEventNotice.isConnected) els.previewStage.appendChild(els.playEventNotice);
  const semanticType = String(entered.semantic_type || "zone");
  const labels = {
    goal: "Entered goal zone",
    hazard: "Entered hazard zone",
    floor_switch: "Activated floor switch",
    target_region: "Entered target region",
  };
  if (state.play.eventTimer !== null) window.clearTimeout(state.play.eventTimer);
  els.playEventNotice.textContent = labels[semanticType] || `Entered ${semanticType.replaceAll("_", " ")}`;
  els.playEventNotice.dataset.kind = semanticType;
  els.playEventNotice.classList.remove("hidden");
  state.play.eventTimer = window.setTimeout(() => {
    state.play.eventTimer = null;
    els.playEventNotice.classList.add("hidden");
  }, 1600);
}

function showGameOverlay(title, detail, failure, action) {
  if (!els.gameOverlay.isConnected) els.previewStage.appendChild(els.gameOverlay);
  els.gameOverlayTitle.textContent = title;
  els.gameOverlayDetail.textContent = detail;
  els.gameOverlayAction.textContent = action;
  els.gameOverlay.classList.toggle("failure", failure);
  els.gameOverlay.classList.remove("hidden");
}

function currentPlayInput() {
  return currentControllerInput(state.play);
}

function currentControllerInput(controlState) {
  const keys = controlState.keys;
  const now = Date.now();
  const active = (code) => keys.has(code) || Number(controlState.keyPulseUntil.get(code) || 0) > now;
  for (const [code, deadline] of controlState.keyPulseUntil) {
    if (deadline <= now) controlState.keyPulseUntil.delete(code);
  }
  return {
    right: Number(active("KeyD") || active("ArrowRight")) - Number(active("KeyA") || active("ArrowLeft")),
    forward: Number(active("KeyW") || active("ArrowUp")) - Number(active("KeyS") || active("ArrowDown")),
    jump: active("Space"),
  };
}

function handlePlayKeyDown(event) {
  const oracle = state.tasks.oracle;
  const controlState = oracle.active ? oracle : state.play.active ? state.play : null;
  if (state.primitiveBrowser.open || !controlState || isEditableTarget(event.target)) return;
  if (event.code === "Escape") {
    event.preventDefault();
    if (oracle.active) void cancelTaskOracle();
    else void stopPlayableEnvironment();
    return;
  }
  if (event.code === "KeyR") {
    event.preventDefault();
    if (!event.repeat) {
      if (oracle.active) void resetTaskOracle();
      else void resetPlayableEnvironment();
    }
    return;
  }
  if (!isPlayControlKey(event.code)) return;
  event.preventDefault();
  controlState.keys.add(event.code);
  controlState.keyPulseUntil.set(event.code, Date.now() + (event.code === "Space" ? 160 : 120));
  if (oracle.active) {
    if (oracle.finishPending || oracle.finishing) return;
    oracle.hasStarted = true;
    oracle.idleTicks = 0;
    oracle.readyToValidate = false;
    if (!oracle.lastTickAt) oracle.lastTickAt = performance.now() - PLAY_TICK_MS;
    scheduleTaskOracleTick(0);
  }
}

function handlePlayKeyUp(event) {
  const controlState = state.tasks.oracle.active ? state.tasks.oracle : state.play.active ? state.play : null;
  if (!controlState || !isPlayControlKey(event.code)) return;
  event.preventDefault();
  controlState.keys.delete(event.code);
}

async function resetPlayableEnvironment() {
  if (!state.play.active || !state.play.sessionId) return;
  state.play.keys.clear();
  state.play.keyPulseUntil.clear();
  if (state.play.resetTimer !== null) window.clearTimeout(state.play.resetTimer);
  if (state.play.eventTimer !== null) window.clearTimeout(state.play.eventTimer);
  state.play.resetTimer = null;
  state.play.eventTimer = null;
  state.play.gameState = "playing";
  els.gameOverlay.classList.add("hidden");
  els.playEventNotice.classList.add("hidden");
  try {
    const response = await fetch(`/api/play/${encodeURIComponent(state.play.sessionId)}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Request failed: ${response.status}`);
    applyPlayState(data.state);
    schedulePlayTick(0);
  } catch (error) {
    appendRevisionMessage("assistant", `Play mode could not reset: ${error.message || String(error)}`);
  }
}

function isPlayMovementKey(code) {
  return ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"].includes(code);
}

function isPlayControlKey(code) {
  return code === "Space" || isPlayMovementKey(code);
}

function activateAgentCamera({ resetOrbit = false, agentId = state.play.agentId } = {}) {
  if (!state.preview.renderer || !agentId) return;
  const camera = state.preview.visualScene?.camera || {};
  if (resetOrbit) state.preview.visual.azimuth = Number(camera.azimuth ?? -42);
  state.preview.visual.elevation = PLAY_CAMERA_ELEVATION;
  state.preview.visual.distance = clamp(
    PLAY_CAMERA_DISTANCE,
    Number(camera.min_distance || 4),
    Number(camera.max_distance || 40),
  );
  state.preview.visual.defaultDistance = state.preview.visual.distance;
  state.preview.visual.panX = 0;
  state.preview.visual.panY = 0;
  state.preview.renderer.beginAgentFollow(agentId);
  els.previewStage.dataset.cameraMode = "agent_follow";
  applyVisualView();
}

function isEditableTarget(target) {
  const tagName = String(target?.tagName || "").toLowerCase();
  return target?.isContentEditable || ["input", "textarea", "select"].includes(tagName);
}

function stopPlayOnPageExit() {
  if (!state.play.sessionId) return;
  const body = new Blob(["{}"], { type: "application/json" });
  navigator.sendBeacon(`/api/play/${encodeURIComponent(state.play.sessionId)}/stop`, body);
}

function mergeScene(scene) {
  const index = state.scenes.findIndex((item) => item.env_id === scene.env_id);
  if (index >= 0) {
    state.scenes.splice(index, 1, scene);
  } else {
    state.scenes.unshift(scene);
  }
}

async function generateEnvironment(payload) {
  state.generating = true;
  setRunning(true);
  setStatus("Planning", "running");
  const activityEvents = [{ type: "request", label: "Request", message: payload.prompt }];
  const streamContext = { payload, activityEvents, assistantNode: null, envId: "" };
  setActivity("Planning", activityEvents);
  try {
    const submittedView = await captureCreateSubmission();
    const requestPayload = withCodexModel({
      ...payload,
      view_context: submittedView.viewContext,
      submitted_view_image: submittedView.imageDataUrl,
    });
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ ...requestPayload, stream: true }),
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    const contentType = response.headers.get("Content-Type") || "";
    const data = contentType.includes("text/event-stream")
      ? await consumeGenerationStream(response, streamContext)
      : await response.json();
    if (!data || data.error || data.isError) throw new Error(data?.error || "Generation failed");
    setActivity(data.status === "success" ? "Finalized" : "Incomplete", data.events || activityEvents);
    if (data.scene) {
      const updatedScene = await fetchScene(data.env_id).catch(() => data.scene);
      mergeScene(updatedScene);
      state.currentScene = updatedScene;
      setStatus(data.status === "success" ? "Finalized" : "Incomplete", data.status === "success" ? "done" : "error");
      if (state.route.page !== "env" || state.route.envId !== data.env_id) {
        openEnvWorkspace(data.env_id);
      } else {
        renderEnvPage();
      }
      setInspectorTab(updatedScene.env_visual_review_pending ? "visual" : updatedScene.env_verification?.has_plan ? "checks" : "objects");
      void ensureVisualReview(updatedScene, data.visual_review_id);
    } else {
      setStatus("Incomplete", "error");
    }
  } catch (error) {
    if (streamContext.assistantNode) appendStreamLine(streamContext.assistantNode, error.message || String(error));
    setStatus("Failed", "error");
    setActivity("Failed", [...activityEvents, { type: "error", label: "Generation failed", message: error.message || String(error) }]);
  } finally {
    state.generating = false;
    setRunning(false);
    if (state.route.page === "env") {
      renderEnvHeader(state.currentScene);
      renderAgentPanelHeader();
    }
  }
}

async function consumeGenerationStream(response, context) {
  return consumeEventStream(response, {
    system(payload) {
      openGenerationWorkspace(payload, context);
    },
    progress(payload) {
      handleGenerationProgress(payload, context);
    },
    scene(payload) {
      handleGenerationScene(payload, context);
    },
    text(payload) {
      if (context.assistantNode) appendStreamLine(context.assistantNode, payload.delta || "");
    },
    done(payload) {
      context.donePayload = payload;
    },
  }).then(() => context.donePayload);
}

function openGenerationWorkspace(payload, context) {
  const envId = String(payload.env_id || "").trim();
  if (!envId || context.envId) return;
  context.envId = envId;
  const provisionalScene = {
    env_id: envId,
    description: context.payload.prompt,
    spec: null,
    visual_scene: null,
    metadata: {},
    objects: [],
    cameras: [],
    previews: {},
    orbit_previews: [],
    history: [{ role: "user", content: context.payload.prompt }],
    status: "generating",
  };
  mergeScene(provisionalScene);
  state.currentScene = provisionalScene;
  openEnvWorkspace(envId);
  setInspectorTab("activity");
  context.assistantNode = appendRevisionMessage("assistant", "Starting generation...");
  setStatus("Building", "running");
  setActivity("Building", context.activityEvents);
}

function openEnvWorkspace(envId) {
  history.pushState(null, "", `#/env/${encodeURIComponent(envId)}`);
  state.route = parseRoute();
  render();
}

function handleGenerationProgress(payload, context) {
  const event = {
    type: payload.type || "progress",
    name: payload.name || "",
    label: payload.label || payload.type || "Progress",
    message: payload.message || "",
  };
  context.activityEvents.push(event);
  setActivity("Building", context.activityEvents);
}

function handleGenerationScene(payload, context) {
  if (!payload.scene?.env_id || payload.scene.env_id !== context.envId) return;
  const liveScene = { ...payload.scene, status: "generating" };
  mergeScene(liveScene);
  state.currentScene = liveScene;
  renderEnvHeader(liveScene);
  renderPreview();
  renderObjects();
  renderEnvChecks(liveScene);
  renderVisualReview(liveScene);
  renderBehaviorTrials(liveScene);
  renderTasks(liveScene);
}

async function reviseEnvironment(message) {
  const scene = state.currentScene;
  if (!scene?.env_id) return;
  const envId = scene.env_id;
  const priorHistory = Array.isArray(scene.history) ? scene.history.slice() : [];
  state.currentScene.history = [...priorHistory, { role: "user", content: message }];
  renderConversation(state.currentScene.history);
  const assistantNode = appendRevisionMessage("assistant", "Starting revision...");
  const activityEvents = [{ type: "request", message }];
  state.revising = true;
  setInspectorTab("activity");
  setRunning(true, "Applying");
  setStatus("Applying", "running");
  setActivity("Applying", activityEvents);
  try {
    const submittedView = await captureRevisionSubmission();
    const response = await fetch(`/api/scenes/${encodeURIComponent(envId)}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(withCodexModel({
        message,
        history: priorHistory,
        view_context: submittedView.viewContext,
        submitted_view_image: submittedView.imageDataUrl,
        stream: true,
      })),
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    const contentType = response.headers.get("Content-Type") || "";
    const data = contentType.includes("text/event-stream")
      ? await consumeRevisionStream(response, { activityEvents, assistantNode })
      : await response.json();
    if (!data || data.error || data.isError) throw new Error(data?.error || "Revision failed");
    setActivity(data.status === "success" ? "Updated" : "Incomplete", data.events || activityEvents);
    if (data.scene) {
      const updatedScene = await fetchScene(envId).catch(() => data.scene);
      mergeScene(updatedScene);
      state.currentScene = updatedScene;
      setStatus(data.status === "success" ? "Updated" : "Incomplete", data.status === "success" ? "done" : "error");
      renderEnvPage();
      setInspectorTab(updatedScene.env_visual_review_pending ? "visual" : updatedScene.env_verification?.has_plan ? "checks" : "objects");
      void ensureVisualReview(updatedScene, data.visual_review_id);
    } else {
      appendStreamLine(assistantNode, "The revision finished without returning an updated scene.");
      setStatus("Incomplete", "error");
    }
  } catch (error) {
    appendStreamLine(assistantNode, error.message || String(error));
    setStatus("Failed", "error");
    setActivity("Failed", [{ type: "error", message: error.message || String(error) }]);
  } finally {
    state.revising = false;
    setRunning(false);
  }
}

async function consumeRevisionStream(response, context) {
  let donePayload = null;
  await consumeEventStream(response, {
    progress(payload) {
      handleRevisionProgress(payload, context);
    },
    text(payload) {
      appendStreamLine(context.assistantNode, payload.delta || "");
    },
    done(payload) {
      donePayload = payload;
    },
  });
  return donePayload;
}

async function ensureVisualReview(scene, preferredReviewId = "", { force = false } = {}) {
  if (!scene?.env_id || scene.status !== "finalized") return;
  let current = scene;
  let pending = current.env_visual_review_pending;
  if (preferredReviewId && pending?.review_id !== preferredReviewId) {
    current = await fetchScene(scene.env_id).catch(() => current);
    pending = current.env_visual_review_pending;
  }
  if (!pending?.review_id || (preferredReviewId && pending.review_id !== preferredReviewId)) return;
  const reviewId = pending.review_id;
  if (state.visualReview.activeIds.has(reviewId)) return;
  if (pending.status === "reviewing") {
    scheduleVisualReviewPoll(current.env_id, reviewId);
    return;
  }
  if (pending.status === "error" && !force) return;
  if (!["awaiting_capture", "evidence_ready", "error"].includes(pending.status)) return;

  state.visualReview.activeIds.add(reviewId);
  recordVisualReviewActivity(current.env_id, "Visual review", pending.status === "awaiting_capture"
    ? "Capturing three styled views."
    : "Retrying saved visual evidence.");
  renderVisualReview(state.currentScene);
  try {
    let evidence = null;
    let action = "rerun";
    if (pending.status === "awaiting_capture") {
      let beforeScene = null;
      if (pending.before) {
        const beforeUrl = pending.before?.visual_scene_url;
        if (!beforeUrl) throw new Error("The before-scene snapshot is missing");
        const beforeResponse = await fetch(beforeUrl, { cache: "no-store" });
        if (!beforeResponse.ok) throw new Error(`Before-scene request failed: ${beforeResponse.status}`);
        beforeScene = await beforeResponse.json();
      }
      evidence = await captureVisualReviewEvidence({
        pending,
        beforeScene,
        afterScene: current.visual_scene,
      });
      action = "evidence";
      recordVisualReviewActivity(current.env_id, "Visual evidence ready", "Three camera views captured without changing the preview.");
    }
    const response = await fetch(
      `/api/scenes/${encodeURIComponent(current.env_id)}/visual-reviews/${encodeURIComponent(reviewId)}/${action}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(withCodexModel(evidence ? { ...evidence, stream: true } : { stream: true })),
      },
    );
    if (!response.ok) throw new Error(`Visual review request failed: ${response.status}`);
    const contentType = response.headers.get("Content-Type") || "";
    const result = contentType.includes("text/event-stream")
      ? await consumeVisualReviewStream(response, current.env_id, reviewId)
      : await response.json();
    if (!result || result.error || result.isError) throw new Error(result?.error || "Visual review failed");
    await refreshSceneAfterVisualReview(current.env_id);
  } catch (error) {
    recordVisualReviewActivity(current.env_id, "Visual review error", error.message || String(error), "error");
    await refreshSceneAfterVisualReview(current.env_id).catch(() => {});
  } finally {
    state.visualReview.activeIds.delete(reviewId);
    renderVisualReview(state.currentScene);
  }
}

async function consumeVisualReviewStream(response, envId, reviewId) {
  let donePayload = null;
  await consumeEventStream(response, {
    visual_review(payload) {
      const status = payload.status || payload.summary?.status || "reviewing";
      const label = status === "reviewing" ? "Reviewing visuals" : "Visual review complete";
      const message = payload.message || payload.summary?.message || "";
      recordVisualReviewActivity(envId, label, message || `Review ${reviewId} is ${status}.`);
      if (state.currentScene?.env_id === envId) {
        if (payload.summary) state.currentScene.env_visual_review = payload.summary;
        if (payload.report) state.currentScene.env_visual_review_report = payload.report;
        renderVisualReview(state.currentScene);
      }
    },
    done(payload) {
      donePayload = payload;
    },
  });
  return donePayload;
}

async function refreshSceneAfterVisualReview(envId) {
  const latest = await fetchScene(envId);
  mergeScene(latest);
  if (state.currentScene?.env_id !== envId) return latest;
  state.currentScene = latest;
  renderEnvHeader(latest);
  renderEnvChecks(latest);
  renderVisualReview(latest);
  renderBehaviorTrials(latest);
  renderTasks(latest);
  renderSceneActivity(latest);
  return latest;
}

async function retryCurrentVisualReview() {
  const scene = state.currentScene;
  if (!scene?.env_id) return;
  let latest = await fetchScene(scene.env_id).catch(() => scene);
  mergeScene(latest);
  state.currentScene = latest;
  const reviewId = latest.env_visual_review_pending?.review_id || "";
  if (!reviewId) return;
  await ensureVisualReview(latest, reviewId, { force: true });
}

async function repairFromVisualReview() {
  const scene = state.currentScene;
  const report = scene?.env_visual_review_report;
  const reviewId = String(report?.review_id || "");
  const failedChecks = (report?.checks || []).filter((check) => check?.passed !== true);
  if (!scene?.env_id || !reviewId || !failedChecks.length || state.running) return;

  const messages = [
    ...new Set(
      failedChecks
        .map((check) => String(check.message || "").replace(/\s+/g, " ").trim())
        .filter(Boolean),
    ),
  ].slice(0, 3);
  const displayMessage = messages.length
    ? `Fix the VLM review issues: ${messages.join("; ")}`.slice(0, 1000)
    : "Fix the issues reported by the VLM review.";
  const priorHistory = Array.isArray(scene.history) ? scene.history.slice() : [];
  state.currentScene.history = [...priorHistory, { role: "user", content: displayMessage }];
  renderConversation(state.currentScene.history);
  const assistantNode = appendRevisionMessage("assistant", "Starting VLM-guided repair...");
  const activityEvents = [{ type: "request", label: "VLM repair", message: displayMessage }];
  state.revising = true;
  setInspectorTab("activity");
  setRunning(true, "Repairing");
  setStatus("Repairing", "running");
  setActivity("Repairing", activityEvents);
  els.visualReviewRepair.disabled = true;
  try {
    const submittedView = await captureRevisionSubmission();
    const response = await fetch(
      `/api/scenes/${encodeURIComponent(scene.env_id)}/visual-reviews/${encodeURIComponent(reviewId)}/repair`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(withCodexModel({
          history: priorHistory,
          view_context: submittedView.viewContext,
          submitted_view_image: submittedView.imageDataUrl,
          stream: true,
        })),
      },
    );
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error || `VLM repair request failed: ${response.status}`);
    }
    const data = await consumeRevisionStream(response, { activityEvents, assistantNode });
    if (!data || data.error || data.isError) throw new Error(data?.error || "VLM-guided repair failed");
    const latest = await fetchScene(scene.env_id).catch(() => data.scene);
    if (!latest) throw new Error("The repair finished without returning an updated scene");
    mergeScene(latest);
    state.currentScene = latest;
    setStatus(data.status === "success" ? "Updated" : "Incomplete", data.status === "success" ? "done" : "error");
    setActivity(data.status === "success" ? "Updated" : "Incomplete", data.events || activityEvents);
    renderEnvPage();
    setInspectorTab(latest.env_visual_review_pending ? "visual" : latest.env_verification?.has_plan ? "checks" : "objects");
    void ensureVisualReview(latest, data.visual_review_id);
  } catch (error) {
    appendStreamLine(assistantNode, error.message || String(error));
    setStatus("Repair failed", "error");
    setActivity("Repair failed", [...activityEvents, { type: "error", message: error.message || String(error) }]);
  } finally {
    state.revising = false;
    setRunning(false);
    renderVisualReview(state.currentScene);
  }
}

function scheduleVisualReviewPoll(envId, reviewId, attempt = 0) {
  if (state.visualReview.pollTimers.has(reviewId)) return;
  const timer = window.setTimeout(async () => {
    state.visualReview.pollTimers.delete(reviewId);
    const latest = await fetchScene(envId).catch(() => null);
    if (!latest) return;
    mergeScene(latest);
    if (state.currentScene?.env_id === envId) {
      state.currentScene = latest;
      renderVisualReview(latest);
    }
    const pending = latest.env_visual_review_pending;
    if (pending?.review_id === reviewId && pending.status === "reviewing" && attempt < 150) {
      scheduleVisualReviewPoll(envId, reviewId, attempt + 1);
    } else if (pending?.review_id === reviewId && pending.status === "reviewing") {
      recordVisualReviewActivity(envId, "Visual review stalled", "The reviewer did not finish; retry is available.", "error");
    }
  }, 2000);
  state.visualReview.pollTimers.set(reviewId, timer);
}

function recordVisualReviewActivity(envId, label, message, type = "visual_review") {
  const current = state.visualReview.activityByEnv.get(envId) || [];
  const event = { type, label, message: String(message || "") };
  if (!current.length || JSON.stringify(current[current.length - 1]) !== JSON.stringify(event)) current.push(event);
  state.visualReview.activityByEnv.set(envId, current.slice(-12));
  if (state.currentScene?.env_id === envId) renderSceneActivity(state.currentScene);
}

async function consumeEventStream(response, handlers) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Streaming response was not readable");
  const decoder = new TextDecoder();
  let buffer = "";
  let streamDone = false;
  const handleEvent = (event) => {
    const payload = parseSsePayload(event.data);
    const handler = handlers[event.type];
    if (handler) handler(payload);
    if (event.type === "done") streamDone = true;
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseChunk(buffer, handleEvent);
    if (streamDone) {
      await reader.cancel();
      return;
    }
  }
  if (buffer.trim()) {
    parseSseChunk(`${buffer}\n\n`, handleEvent);
  }
}

function parseSseChunk(buffer, onEvent) {
  const chunks = buffer.split(/\n\n/);
  const rest = chunks.pop() || "";
  for (const chunk of chunks) {
    const event = { type: "message", data: "" };
    const dataLines = [];
    for (const line of chunk.split(/\n/)) {
      if (line.startsWith("event:")) {
        event.type = line.slice("event:".length).trim() || "message";
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trimStart());
      }
    }
    event.data = dataLines.join("\n");
    onEvent(event);
  }
  return rest;
}

function parseSsePayload(data) {
  if (!data) return {};
  try {
    return JSON.parse(data);
  } catch {
    return { message: data };
  }
}

function handleRevisionProgress(payload, context) {
  const label = payload.label || payload.type || "Progress";
  const message = payload.message || "";
  context.activityEvents.push({ type: payload.type || "progress", name: payload.name || "", label, message });
  setActivity("Applying", context.activityEvents);
}

function appendStreamLine(node, text) {
  const value = String(text || "").trim();
  if (!value) return;
  const current = String(node.dataset.rawContent || node.textContent || "").trim();
  const isPlaceholder = current === "Starting revision..." || current === "Starting generation...";
  const nextContent = current && !isPlaceholder
    ? `${current}\n${value}`
    : value;
  setConversationContent(node, "assistant", nextContent);
  els.revisionMessages.scrollTop = els.revisionMessages.scrollHeight;
}

function captureUserViewContext() {
  const rect = els.previewStage.getBoundingClientRect();
  const visual = state.preview.visual;
  const visualBasis = horizontalScreenBasis(Number(visual.azimuth || 0));
  const activeBasis = activePreviewScreenBasis();
  return {
    version: 1,
    capture_kind: "structured_before_edit",
    captured_at: new Date().toISOString(),
    route: state.route.page,
    preview_mode: state.preview.mode,
    selected_object_id: null,
    scene: {
      env_id: state.currentScene?.env_id || null,
      object_count: (state.currentScene?.objects || []).length,
      object_ids: (state.currentScene?.objects || []).map((object) => object.id).filter(Boolean),
    },
    viewport: {
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      device_pixel_ratio: roundNumber(window.devicePixelRatio || 1),
    },
    coordinate_conventions: {
      world_right: "+x",
      world_left: "-x",
      world_forward: "+y",
      world_back: "-y",
      world_up: "+z",
      default_unqualified_left_right: "screen_space_current_preview",
    },
    screen_space: activeBasis,
    visual: {
      rendered: Boolean(state.preview.renderer),
      azimuth_degrees: roundNumber(visual.azimuth),
      elevation_degrees: roundNumber(visual.elevation),
      distance: roundNumber(visual.distance),
      default_distance: roundNumber(visual.defaultDistance),
      zoom_percent: roundNumber((Number(visual.defaultDistance || 1) / Number(visual.distance || 1)) * 100),
      pan_x_world: roundNumber(visual.panX),
      pan_z_world: roundNumber(visual.panY),
      target: roundVector(state.preview.visualScene?.camera?.target || []),
      screen_right_world_xy: visualBasis.screen_right_world_xy,
      screen_left_world_xy: visualBasis.screen_left_world_xy,
    },
    physics_debug: {
      zoom: roundNumber(state.preview.zoom),
      pan_x_pixels: roundNumber(state.preview.x),
      pan_y_pixels: roundNumber(state.preview.y),
      orbit_index: state.preview.orbitIndex,
      orbit_frame_count: state.preview.orbitFrames.length,
      orbit_azimuth_degrees: activeBasis.source === "mujoco_orbit_frame" ? activeBasis.azimuth_degrees : null,
    },
  };
}

async function captureRevisionSubmission() {
  const viewContext = captureUserViewContext();
  if (state.preview.mode !== "visual" || !state.preview.visualScene) {
    return { viewContext, imageDataUrl: "" };
  }
  const rect = els.previewStage.getBoundingClientRect();
  const visual = state.preview.visual;
  const capture = await captureSubmittedView({
    visualScene: state.preview.visualScene,
    physicsObjects: state.currentScene?.objects || [],
    view: {
      target: state.preview.visualScene?.camera?.target || [0, 0, 0.5],
      distance: visual.distance,
      azimuth: visual.azimuth,
      elevation: visual.elevation,
      panX: visual.panX,
      panY: visual.panY,
    },
    viewport: { width: rect.width, height: rect.height },
  });
  viewContext.version = 2;
  viewContext.submitted_image = capture.image;
  viewContext.screen_space = {
    ...viewContext.screen_space,
    ...capture.screen_space,
    source: "threejs_visual_camera",
    reliable: true,
  };
  viewContext.visual = {
    ...viewContext.visual,
    camera: capture.screen_space.camera,
    playable_area: capture.screen_space.playable_area,
    regions: capture.screen_space.regions,
    projected_objects: capture.screen_space.projected_objects,
  };
  return { viewContext, imageDataUrl: capture.image_data_url };
}

async function captureCreateSubmission() {
  if (!state.createPreview.renderer || !state.createPreview.ready) {
    return { viewContext: {}, imageDataUrl: "" };
  }
  await state.createPreview.ready;
  const visualScene = state.createPreview.visualScene;
  const view = state.createPreview.view;
  if (!visualScene || !view || !state.createPreview.renderer) {
    return { viewContext: {}, imageDataUrl: "" };
  }
  const rect = els.createPreviewStage.getBoundingClientRect();
  const objects = Array.isArray(state.createPreview.spec?.objects) ? state.createPreview.spec.objects : [];
  const capture = await captureSubmittedView({
    visualScene,
    physicsObjects: objects,
    view,
    viewport: { width: rect.width, height: rect.height },
  });
  const visualBasis = horizontalScreenBasis(Number(view.azimuth || 0));
  return {
    imageDataUrl: capture.image_data_url,
    viewContext: {
      version: 2,
      capture_kind: "structured_before_generation",
      captured_at: new Date().toISOString(),
      route: "create",
      preview_mode: "visual",
      selected_object_id: null,
      scene: {
        env_id: null,
        baseline: "empty_courtyard",
        object_count: objects.length,
        object_ids: objects.map((object) => object.id).filter(Boolean),
      },
      viewport: {
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        device_pixel_ratio: roundNumber(window.devicePixelRatio || 1),
      },
      coordinate_conventions: {
        world_right: "+x",
        world_left: "-x",
        world_forward: "+y",
        world_back: "-y",
        world_up: "+z",
        default_unqualified_left_right: "screen_space_current_preview",
      },
      screen_space: {
        ...capture.screen_space,
        source: "threejs_visual_camera",
        reliable: true,
      },
      visual: {
        rendered: true,
        azimuth_degrees: roundNumber(view.azimuth),
        elevation_degrees: roundNumber(view.elevation),
        distance: roundNumber(view.distance),
        default_distance: roundNumber(view.distance),
        zoom_percent: 100,
        pan_x_world: roundNumber(view.panX),
        pan_z_world: roundNumber(view.panY),
        target: roundVector(view.target),
        screen_right_world_xy: visualBasis.screen_right_world_xy,
        screen_left_world_xy: visualBasis.screen_left_world_xy,
        camera: capture.screen_space.camera,
        playable_area: capture.screen_space.playable_area,
        regions: capture.screen_space.regions,
        projected_objects: capture.screen_space.projected_objects,
      },
    },
  };
}

function activePreviewScreenBasis() {
  if (state.preview.mode === "physics" && state.preview.orbitFrames.length) {
    const frameCount = state.preview.orbitFrames.length;
    const frameIndex = clamp(state.preview.orbitIndex, 0, frameCount - 1);
    const azimuth = wrapDegrees(-180 + 360 * frameIndex / frameCount);
    return {
      ...horizontalScreenBasis(azimuth),
      source: "mujoco_orbit_frame",
      frame_index: frameIndex,
      frame_count: frameCount,
      reliable: true,
    };
  }
  if (state.preview.mode === "physics") {
    return {
      ...horizontalScreenBasis(0),
      source: "canonical_fallback_no_orbit",
      reliable: false,
    };
  }
  return {
    ...horizontalScreenBasis(Number(state.preview.visual.azimuth || 0)),
    source: "threejs_visual_camera",
    reliable: true,
  };
}

function horizontalScreenBasis(azimuthDegrees) {
  const azimuthRadians = Number(azimuthDegrees || 0) * Math.PI / 180;
  const screenRightWorldXY = [
    roundNumber(Math.cos(azimuthRadians)),
    roundNumber(-Math.sin(azimuthRadians)),
  ];
  return {
    azimuth_degrees: roundNumber(wrapDegrees(azimuthDegrees)),
    screen_right_world_xy: screenRightWorldXY,
    screen_left_world_xy: screenRightWorldXY.map((value) => roundNumber(-value)),
  };
}

function parseRoute() {
  const raw = window.location.hash.replace(/^#/, "") || "/";
  if (raw === "/create") return { page: "create" };
  if (raw.startsWith("/env/")) {
    return { page: "env", envId: decodeURIComponent(raw.slice("/env/".length)) };
  }
  return { page: "home" };
}

function navigateHome() {
  window.location.hash = "/";
}

function navigateCreate() {
  if (state.route.page !== "create") {
    els.generateForm.reset();
    setEnvNameValidation();
  }
  window.location.hash = "/create";
}

function navigateEnv(envId) {
  window.location.hash = `/env/${encodeURIComponent(envId)}`;
}

async function createVariation() {
  const scene = state.currentScene;
  if (!scene?.env_id || state.running) return;
  const generation = scene.generation || scene.spec?.generation || {};
  const originalRequest = (scene.history || []).find((turn) => turn.role === "user")?.content || scene.description;
  await generateEnvironment({
    name: `${sceneDisplayName(scene)} variation`,
    prompt: originalRequest || "Create an interesting robot courtyard level.",
    family: generation.family || "",
    difficulty: generation.difficulty || "medium",
    seed: randomLevelSeed(),
  });
}

function randomLevelSeed() {
  const values = new Uint32Array(2);
  window.crypto.getRandomValues(values);
  return Number((BigInt(values[0]) << 21n) | BigInt(values[1] & 0x1fffff));
}

function render() {
  els.sceneCount.textContent = `${state.scenes.length} ${state.scenes.length === 1 ? "environment" : "environments"}`;
  renderTopStatus();
  if (state.route.page === "create") {
    showPage("create");
    disposeVisualRenderer();
    renderCreatePage();
    return;
  }
  if (state.route.page === "env") {
    disposeCreatePreviewRenderer();
    showPage("env");
    state.currentScene = state.scenes.find((scene) => scene.env_id === state.route.envId) || null;
    renderEnvPage();
    return;
  }
  showPage("home");
  disposeVisualRenderer();
  disposeCreatePreviewRenderer();
  renderHomePage();
}

function showPage(page) {
  document.body.dataset.page = page;
  els.homePage.classList.toggle("hidden", page !== "home");
  els.createPage.classList.toggle("hidden", page !== "create");
  els.envPage.classList.toggle("hidden", page !== "env");
  els.newEnvButton.classList.toggle("hidden", page !== "env");
}

function renderTopStatus() {
  if (state.running) return;
  if (state.route.page === "env" && state.currentScene) {
    setStatus(sceneStatusLabel(state.currentScene), sceneStatusClass(state.currentScene));
  } else {
    els.topStatus.className = "status hidden";
  }
}

function renderHomePage() {
  const hasScenes = state.scenes.length > 0;
  els.sceneGrid.replaceChildren();
  els.homePage.classList.toggle("is-empty", !hasScenes);
  els.homeCreateButton.classList.toggle("hidden", !hasScenes);
  els.emptyLibrary.classList.toggle("hidden", hasScenes);
  if (!hasScenes) return;
  for (const scene of state.scenes) {
    els.sceneGrid.appendChild(sceneCard(scene));
  }
}

function sceneCard(scene) {
  const card = document.createElement("article");
  card.className = "scene-card";

  const openButton = document.createElement("button");
  openButton.type = "button";
  openButton.className = "scene-card-open";
  openButton.setAttribute("aria-label", `Open ${sceneDisplayName(scene)}`);
  openButton.addEventListener("click", () => navigateEnv(scene.env_id));

  const thumb = document.createElement("div");
  thumb.className = "scene-thumb";
  const previewUrl = scene.styled_preview_url || "";
  if (previewUrl) {
    const img = document.createElement("img");
    img.src = previewUrl;
    img.alt = `${sceneDisplayName(scene)} styled preview`;
    img.loading = "lazy";
    img.decoding = "async";
    img.addEventListener("error", () => {
      thumb.replaceChildren();
      appendText(thumb, "div", "thumb-empty", "Styled preview unavailable");
    }, { once: true });
    thumb.appendChild(img);
  } else {
    const reviewStatus = String(scene.env_visual_review?.status || "");
    const pending = ["pending_generation", "awaiting_capture", "evidence_ready", "reviewing"].includes(reviewStatus);
    appendText(thumb, "div", "thumb-empty", pending ? "Preparing styled preview" : "Styled preview unavailable");
  }

  const meta = document.createElement("div");
  meta.className = "scene-meta";
  const title = document.createElement("strong");
  title.textContent = sceneDisplayName(scene);
  const description = document.createElement("span");
  description.textContent = scene.description || "3D environment";
  const badges = document.createElement("div");
  badges.className = "scene-badges";
  badges.append(
    badge(sceneStatusLabel(scene), scene.status === "finalized" ? "ready" : ""),
    badge(`${(scene.objects || []).length} objects`, "quiet"),
  );
  const generation = scene.generation || scene.spec?.generation;
  if (generation?.family) {
    badges.append(badge(String(generation.family).replaceAll("_", " "), "quiet"));
  }
  const checks = scene.env_verification;
  if (checks?.has_plan) {
    badges.append(badge(checks.label || "Env checks", checks.status === "passed" ? "ready" : "warn"));
  }
  const visualReview = scene.env_visual_review;
  if (visualReview && visualReview.status !== "not_run") {
    const reviewClass = visualReview.status === "passed" ? "ready" : "warn";
    badges.append(badge(visualReview.label || "Visual review", reviewClass));
  }
  meta.append(title, description, badges);
  openButton.append(thumb, meta);

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "scene-card-delete";
  deleteButton.textContent = "Delete";
  deleteButton.setAttribute("aria-label", `Delete ${sceneDisplayName(scene)}`);
  deleteButton.addEventListener("click", () => void deleteEnvironment(scene, deleteButton));

  card.append(openButton, deleteButton);
  return card;
}

async function deleteEnvironment(scene, button) {
  const name = sceneDisplayName(scene);
  const confirmed = window.confirm(
    `Delete “${name}”? This permanently removes its tasks, tests, and saved runs.`,
  );
  if (!confirmed) return;

  const card = button.closest(".scene-card");
  button.disabled = true;
  button.textContent = "Deleting...";
  card?.classList.add("is-deleting");
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(scene.env_id)}`, { method: "DELETE" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) throw new Error(data.error || `Delete failed: ${response.status}`);
    state.scenes = state.scenes.filter((item) => item.env_id !== scene.env_id);
    if (state.currentScene?.env_id === scene.env_id) state.currentScene = null;
    render();
    setStatus("Environment deleted", "ready");
  } catch (error) {
    card?.classList.remove("is-deleting");
    button.disabled = false;
    button.textContent = "Delete";
    setStatus("Delete failed", "error");
    window.alert(error.message || String(error));
  }
}

function sceneDisplayName(scene) {
  const displayName = String(scene?.display_name || "").trim();
  return displayName || String(scene?.env_id || "Untitled environment");
}

function badge(label, className) {
  const node = document.createElement("span");
  node.className = `scene-badge ${className}`.trim();
  node.textContent = label;
  return node;
}

function renderCreatePage() {
  ensureCreatePreview();
}

function setEnvNameValidation(message = "") {
  const hasError = Boolean(message);
  els.envName.setAttribute("aria-invalid", String(hasError));
  els.envNameError.textContent = message;
  els.envNameError.classList.toggle("hidden", !hasError);
}

function renderEnvPage() {
  const scene = state.currentScene;
  renderEnvHeader(scene);
  renderAgentPanelHeader();
  renderPreview();
  renderConversation(scene?.history || []);
  renderEnvChecks(scene);
  renderVisualReview(scene);
  renderBehaviorTrials(scene);
  renderTasks(scene);
  renderObjects();
  renderSceneActivity(scene);
  if (scene?.status === "finalized" && scene.env_visual_review_pending) {
    queueMicrotask(() => void ensureVisualReview(scene));
  }
}

function renderEnvHeader(scene) {
  if (!scene) {
    els.activeTitle.textContent = "Environment not found";
    els.sceneLabel.textContent = "Return home and select an environment.";
    els.revisionPrompt.disabled = true;
    els.revisionButton.disabled = true;
    updatePlayButton(null);
    els.levelMeta.replaceChildren();
    els.levelMeta.classList.add("hidden");
    els.variationButton.classList.add("hidden");
    return;
  }
  els.activeTitle.textContent = sceneDisplayName(scene);
  const objectCount = (scene.objects || []).length;
  const parts = [`${objectCount} ${objectCount === 1 ? "object" : "objects"}`];
  const checkSummary = scene.env_verification_report?.summary;
  if (checkSummary?.total) {
    const failures = Number(checkSummary.critical_failures || 0) + Number(checkSummary.advisory_failures || 0);
    parts.push(failures ? `${failures} check issues` : `${checkSummary.passed} checks passed`);
  }
  const visualReview = scene.env_visual_review;
  if (visualReview?.status === "reviewing" || visualReview?.status === "awaiting_capture") {
    parts.push("visual review running");
  } else if (visualReview?.status === "needs_attention") {
    parts.push("visual review needs attention");
  } else if (visualReview?.status === "passed") {
    parts.push("visual review passed");
  }
  const behavior = scene.env_behavior_trials;
  if (behavior?.status === "running" || hasServerActiveBehaviorRuns(behavior)) {
    parts.push("agent test running");
  } else if (behavior?.status === "failed" || behavior?.status === "needs_attention") {
    parts.push("agent test needs attention");
  } else if (behavior?.status === "passed") {
    parts.push("agent test passed");
  }
  if (state.generating) parts.push("building live");
  els.sceneLabel.textContent = parts.join(" / ");
  const generation = scene.generation || scene.spec?.generation;
  els.levelMeta.replaceChildren();
  els.levelMeta.classList.toggle("hidden", !generation?.family);
  if (generation?.family) {
    appendText(els.levelMeta, "span", "", generation.family.replaceAll("_", " "));
  }
  els.variationButton.classList.toggle("hidden", !generation);
  const taskBusy = state.tasks.compiling || state.tasks.oracle.active;
  els.variationButton.disabled = state.running || state.play.active || taskBusy;
  els.revisionPrompt.disabled = state.running || state.play.active || taskBusy;
  els.revisionButton.disabled = state.running || state.play.active || taskBusy;
  updatePlayButton(scene);
  if (!state.running && !state.play.active && !taskBusy) setStatus(sceneStatusLabel(scene), sceneStatusClass(scene));
}

function updatePlayButton(scene) {
  if (!els.playEnvButton) return;
  const isCurrentPlay = Boolean(state.play.active && scene?.env_id === state.play.envId);
  const playable = Boolean(scene?.capabilities?.playable ?? (scene?.objects || []).some((item) => item.semantic_type === "agent"));
  els.playEnvButton.disabled = state.launchingPlay || state.tasks.oracle.active || state.tasks.compiling || (!isCurrentPlay && (!scene || scene.status !== "finalized" || state.running || !playable));
  els.playEnvButton.classList.toggle("playing", isCurrentPlay);
  els.playEnvButton.title = playable ? "Play the authored MuJoCo agent" : "This environment has no authored agent";
  els.playEnvButton.textContent = state.launchingPlay
    ? "Starting..."
    : isCurrentPlay
      ? "Stop"
      : playable
        ? "Play"
        : "No Agent";
}

function updatePlayModeControls() {
  const controlMode = state.tasks.oracle.active ? "oracle" : state.play.active ? "play" : "";
  if (controlMode && !els.movementHelp.isConnected) els.previewStage.appendChild(els.movementHelp);
  els.previewStage.classList.toggle("is-playing", state.play.active);
  const taskBusy = state.tasks.oracle.active || state.tasks.compiling;
  els.previewStage.classList.toggle("is-recording-oracle", state.tasks.oracle.active);
  els.movementHelp.classList.toggle("hidden", !controlMode);
  els.movementHelp.dataset.mode = controlMode;
  els.movementHelpMode.textContent = controlMode === "oracle" ? "Oracle controls" : "Play controls";
  els.movementHelpExit.textContent = controlMode === "oracle" ? "Cancel" : "Exit";
  els.revisionPrompt.disabled = state.running || state.play.active || taskBusy || !state.currentScene;
  els.revisionButton.disabled = state.running || state.play.active || taskBusy || !state.currentScene;
}

function renderAgentPanelHeader() {
  if (!els.conversationTitle || !els.conversationSubtitle) return;
  updateRevisionComposerState();
  if (state.generating) {
    els.conversationTitle.textContent = "Building Environment";
    els.conversationSubtitle.textContent = "Updates appear here as the scene takes shape.";
    setAgentState("Building", "running");
    return;
  }
  if (state.revising) {
    els.conversationTitle.textContent = "Updating Environment";
    els.conversationSubtitle.textContent = "Applying your latest changes.";
    setAgentState("Working", "running");
    return;
  }
  els.conversationTitle.textContent = "Edit With Agent";
  els.conversationSubtitle.textContent = "Describe a change in natural language.";
  setAgentState("Ready", "ready");
}

function updateRevisionComposerState() {
  const busy = Boolean(state.running && state.currentScene);
  const busyLabel = state.revising ? "Applying" : state.generating ? "Building" : "Working";
  els.revisionForm.classList.toggle("is-busy", busy);
  els.revisionButton.classList.toggle("is-busy", busy);
  els.revisionButton.setAttribute("aria-busy", busy ? "true" : "false");
  els.revisionButton.textContent = busy ? busyLabel : "Apply";
  els.revisionPrompt.placeholder = busy
    ? state.revising
      ? "Codex is applying your change..."
      : "Codex is building the environment..."
    : "Describe the next change...";
}

function setAgentState(label, className) {
  if (!els.agentState) return;
  els.agentState.textContent = label;
  els.agentState.className = `agent-state ${className || ""}`.trim();
}

function setInspectorTab(tabName) {
  const allowed = new Set(["checks", "visual", "behavior", "tasks", "objects", "activity"]);
  const nextTab = allowed.has(tabName) ? tabName : "checks";
  state.inspectorTab = nextTab;
  for (const button of document.querySelectorAll("[data-inspector-tab]")) {
    const active = button.dataset.inspectorTab === nextTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.tabIndex = active ? 0 : -1;
  }
  for (const pane of document.querySelectorAll("[data-inspector-pane]")) {
    const active = pane.dataset.inspectorPane === nextTab;
    pane.classList.toggle("active", active);
    pane.classList.toggle("hidden", !active);
  }
}

function showInspectorTooltip(anchor) {
  const message = String(anchor?.dataset?.tooltip || "").trim();
  if (!message || !els.inspectorTooltip) return;

  const tooltip = els.inspectorTooltip;
  tooltip.textContent = message;
  tooltip.hidden = false;
  tooltip.style.visibility = "hidden";
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";

  const anchorRect = anchor.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const viewportPadding = 12;
  const gap = 10;
  const anchorCenter = anchorRect.left + anchorRect.width / 2;
  const maxLeft = Math.max(viewportPadding, window.innerWidth - tooltipRect.width - viewportPadding);
  const left = Math.min(maxLeft, Math.max(viewportPadding, anchorCenter - tooltipRect.width / 2));
  const top = Math.max(viewportPadding, anchorRect.top - tooltipRect.height - gap);
  const arrowLeft = Math.min(tooltipRect.width - 16, Math.max(16, anchorCenter - left));

  tooltip.style.left = `${Math.round(left)}px`;
  tooltip.style.top = `${Math.round(top)}px`;
  tooltip.style.setProperty("--tooltip-arrow-left", `${Math.round(arrowLeft)}px`);
  tooltip.style.visibility = "visible";
}

function hideInspectorTooltip() {
  if (els.inspectorTooltip) els.inspectorTooltip.hidden = true;
}

function renderPreview() {
  const scene = state.currentScene;
  const nextSceneId = scene?.env_id || "";
  const nextSceneVersion = previewVersionKey(scene);
  if (nextSceneId !== state.preview.sceneId || nextSceneVersion !== state.preview.sceneVersion) {
    resetPreviewState(nextSceneId, nextSceneVersion);
  }
  return renderVisualPreview(scene);
}

function renderVisualPreview(scene) {
  disposeVisualRenderer();
  els.previewStage.replaceChildren();
  els.previewStage.classList.remove("has-preview", "is-panning");
  els.previewStage.classList.add("has-visual");
  const visualScene = scene?.visual_scene;
  if (!visualScene) {
    if (state.generating) {
      const loading = document.createElement("div");
      loading.className = "preview-loading";
      const blocks = document.createElement("span");
      blocks.className = "preview-loading-blocks";
      blocks.setAttribute("aria-hidden", "true");
      blocks.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
      const label = document.createElement("strong");
      label.textContent = "Preparing the live preview";
      const detail = document.createElement("span");
      detail.textContent = "Objects will appear as Codex adds them.";
      loading.append(blocks, label, detail);
      els.previewStage.appendChild(loading);
    } else {
      appendText(els.previewStage, "div", "empty-preview", "Visual preview not available.");
    }
    state.preview.ready = Promise.resolve(false);
    return state.preview.ready;
  }
  const host = document.createElement("div");
  host.className = "visual-preview";
  els.previewStage.appendChild(host);
  try {
    state.preview.renderer = new VisualPreviewRenderer(host);
    const renderer = state.preview.renderer;
    state.preview.visualScene = visualScene;
    state.preview.ready = renderer.setScene(visualScene).then(() => {
      if (state.preview.renderer !== renderer) return;
      applyVisualView();
      return true;
    }).catch((error) => {
      if (state.preview.renderer !== renderer) return;
      disposeVisualRenderer();
      appendText(els.previewStage, "div", "empty-preview", `Visual preview failed: ${error.message || error}`);
      return false;
    });
  } catch (error) {
    disposeVisualRenderer();
    appendText(els.previewStage, "div", "empty-preview", `Visual preview failed: ${error.message || error}`);
    state.preview.ready = Promise.resolve(false);
    return state.preview.ready;
  }
  if (!state.preview.sceneId || scene.env_id !== state.preview.sceneId) {
    resetVisualViewFromScene(visualScene);
  }
  applyVisualView();
  return state.preview.ready;
}

function renderPhysicsPreview(scene) {
  disposeVisualRenderer();
  els.previewStage.replaceChildren();
  els.previewStage.classList.remove("has-visual", "has-preview", "is-panning");
  state.preview.orbitFrames = orbitFramesFor(scene);
  state.preview.orbitIndex = clamp(Math.round(state.preview.orbitIndex), 0, Math.max(0, state.preview.orbitFrames.length - 1));
  const url = physicsPreviewUrlFor(scene);
  if (!url) {
    state.preview.url = "";
    appendText(els.previewStage, "div", "empty-preview", "Preview not rendered.");
    return;
  }
  state.preview.url = url;
  const img = document.createElement("img");
  img.className = "preview-image";
  img.src = url;
  img.alt = `${sceneDisplayName(scene)} physics preview`;
  img.draggable = false;
  els.previewStage.classList.add("has-preview");
  els.previewStage.appendChild(img);
  updatePhysicsTransform();
}

function renderObjects() {
  const objects = state.currentScene?.objects || [];
  els.objectSummary.textContent = `${objects.length} ${objects.length === 1 ? "object" : "objects"}`;
  els.objectTabBadge.textContent = String(objects.length);
  els.objectList.replaceChildren();
  if (!objects.length) {
    appendText(els.objectList, "div", "empty-list", "No objects.");
    return;
  }
  const header = document.createElement("div");
  header.className = "object-row object-row-head";
  for (const label of ["Object", "Type", "Position", "Size"]) {
    appendText(header, "span", "", label);
  }
  els.objectList.appendChild(header);
  for (const object of objects) {
    const row = document.createElement("div");
    row.className = "object-row";
    const name = document.createElement("strong");
    name.textContent = object.id || "object";
    const semantic = document.createElement("span");
    semantic.dataset.label = "Type";
    semantic.textContent = object.semantic_type || "";
    const position = document.createElement("span");
    position.dataset.label = "Position";
    position.textContent = Array.isArray(object.position) ? object.position.map(formatNumber).join(", ") : "";
    const size = document.createElement("span");
    size.dataset.label = "Size";
    size.textContent = Array.isArray(object.size) ? object.size.map(formatNumber).join(" x ") : "";
    row.append(name, semantic, position, size);
    els.objectList.appendChild(row);
  }
}

function renderEnvChecks(scene) {
  els.envCheckList.replaceChildren();
  els.envCheckStats.replaceChildren();
  const summary = scene?.env_verification;
  const report = scene?.env_verification_report;
  const plan = scene?.env_verification_plan;
  if (!scene || !summary?.has_plan) {
    els.envCheckSummary.textContent = "Not defined";
    els.checkTabBadge.textContent = "0";
    els.checkTabBadge.className = "tab-count";
    appendText(els.envCheckList, "div", "empty-list", "No deterministic checks defined.");
    return;
  }
  const reportSummary = report?.summary || {};
  const total = Number(reportSummary.total || plan?.checks?.length || 0);
  const passedCount = Number(reportSummary.passed || 0);
  const criticalFailures = Number(reportSummary.critical_failures || 0);
  const advisoryFailures = Number(reportSummary.advisory_failures || 0);
  const issueCount = criticalFailures + advisoryFailures;
  els.envCheckSummary.textContent = report
    ? issueCount ? `${issueCount} ${issueCount === 1 ? "issue" : "issues"} need attention` : "All deterministic checks passed"
    : "Checks are defined and waiting to run";
  els.checkTabBadge.textContent = report ? `${passedCount}/${total}` : String(total);
  els.checkTabBadge.className = `tab-count ${report ? issueCount ? "bad" : "good" : ""}`.trim();
  if (report) {
    appendCheckStat("Passed", passedCount, "good");
    appendCheckStat("Critical", criticalFailures, criticalFailures ? "bad" : "quiet");
    appendCheckStat("Advisory", advisoryFailures, advisoryFailures ? "warn" : "quiet");
  }
  const checks = Array.isArray(report?.results) ? report.results : Array.isArray(plan?.checks) ? plan.checks : [];
  if (!checks.length) {
    appendText(els.envCheckList, "div", "empty-list", summary.has_report ? "No checks reported." : "Checks have not run.");
    return;
  }
  for (const check of checks) {
    const item = document.createElement("div");
    const hasResult = typeof check.passed === "boolean" || ["pass", "fail"].includes(String(check.status || ""));
    const passed = hasResult && (check.passed === true || check.status === "pass");
    const severity = String(check.severity || "critical").toLowerCase();
    const resultClass = !hasResult ? "pending" : passed ? "passed" : "failed";
    item.className = `env-check-item ${resultClass} ${severity}`.trim();

    const top = document.createElement("div");
    top.className = "env-check-top";
    const indicator = document.createElement("span");
    indicator.className = `check-indicator ${resultClass}`;
    indicator.setAttribute("aria-hidden", "true");
    const name = document.createElement("strong");
    name.textContent = check.description || check.id || check.type || "check";
    const pill = document.createElement("span");
    pill.className = "env-check-pill";
    pill.textContent = !hasResult ? "pending" : passed ? "passed" : severity === "advisory" ? "advisory" : "failed";
    const title = document.createElement("div");
    title.className = "env-check-title";
    title.append(indicator, name);
    top.append(title, pill);
    item.appendChild(top);

    if (check.message) {
      appendText(item, "div", "env-check-message", check.message);
    }
    const hints = Array.isArray(check.repair_hints) ? check.repair_hints.filter(Boolean) : [];
    if (hints.length) {
      appendText(item, "div", "env-check-hint", hints.join(" "));
    }
    const metrics = compactMetrics(check.metrics);
    if (metrics) {
      const details = document.createElement("details");
      details.className = "env-check-details";
      const summaryNode = document.createElement("summary");
      summaryNode.textContent = "View metrics";
      const metricsNode = document.createElement("pre");
      metricsNode.className = "env-check-metrics";
      metricsNode.textContent = metrics;
      details.append(summaryNode, metricsNode);
      item.appendChild(details);
    }
    els.envCheckList.appendChild(item);
  }
}

function renderVisualReview(scene) {
  els.visualReviewStats.replaceChildren();
  els.visualReviewBody.replaceChildren();
  const summary = scene?.env_visual_review || { status: "not_run", label: "Visual review: not run" };
  const report = scene?.env_visual_review_report;
  const pending = scene?.env_visual_review_pending;
  const active = Boolean(pending?.review_id && state.visualReview.activeIds.has(pending.review_id));
  const failedChecks = Array.isArray(report?.checks) ? report.checks.filter((check) => check?.passed !== true) : [];
  const reportIsCurrent = Boolean(report?.review_id && report.review_id === summary.review_id && !summary.stale);
  const repairable = !active
    && !state.running
    && reportIsCurrent
    && failedChecks.length > 0
    && ["passed", "needs_attention"].includes(String(summary.status || ""));
  updateVisualTabBadge(summary, report);
  els.visualReviewSummary.textContent = String(summary.label || `Visual review: ${summary.status || "not run"}`).replace(/^Visual review:\s*/i, "");
  const retryable = !active && (
    ["awaiting_capture", "evidence_ready", "error"].includes(String(pending?.status || ""))
    || String(summary.status || "") === "error"
  );
  els.visualReviewRetry.classList.toggle("hidden", !retryable);
  els.visualReviewRetry.disabled = active;
  els.visualReviewRepair.classList.toggle("hidden", !repairable);
  els.visualReviewRepair.disabled = !repairable;

  if (["pending_generation", "awaiting_capture", "evidence_ready", "reviewing"].includes(summary.status)) {
    appendVisualReviewStatus(
      summary.status === "reviewing" ? "Reviewing styled views" : "Preparing visual evidence",
      summary.status === "reviewing"
        ? "The environment remains editable and playable while the review runs."
        : "Three consistent camera views will be captured from the authored scene.",
      "running",
    );
    if (!report || report.review_id !== pending?.review_id) return;
  } else if (!report) {
    appendVisualReviewStatus(
      "No visual review yet",
      "Studio automatically reviews styled views after a successful generation or revision.",
      "quiet",
    );
    return;
  }

  const reportSummary = report?.summary || {};
  if (report) {
    appendVisualReviewStat("Passed", Number(reportSummary.passed || 0), "good");
    appendVisualReviewStat("Critical", Number(reportSummary.critical_failures || 0), Number(reportSummary.critical_failures || 0) ? "bad" : "quiet");
    appendVisualReviewStat("Advisory", Number(reportSummary.advisory_failures || 0), Number(reportSummary.advisory_failures || 0) ? "warn" : "quiet");
    if (reportSummary.message) appendText(els.visualReviewBody, "p", "visual-review-summary", reportSummary.message);
    renderVisualReviewGallery(report);
    renderVisualReviewChecks(report);
    renderVisualReviewDetails(report);
  }
}

function appendVisualReviewStatus(title, message, className) {
  const status = document.createElement("div");
  status.className = `visual-review-status ${className || ""}`.trim();
  if (className === "running") {
    const head = document.createElement("div");
    head.className = "visual-review-status-head";
    appendText(head, "strong", "", title);
    const progress = document.createElement("span");
    progress.className = "visual-review-status-progress";
    renderProgressDots(progress, "Visual review running");
    head.appendChild(progress);
    status.appendChild(head);
  } else {
    appendText(status, "strong", "", title);
  }
  appendText(status, "span", "", message);
  els.visualReviewBody.appendChild(status);
}

function appendVisualReviewStat(label, value, className) {
  const item = document.createElement("div");
  item.className = `check-stat ${className || ""}`.trim();
  appendText(item, "strong", "", String(value));
  appendText(item, "span", "", label);
  els.visualReviewStats.appendChild(item);
}

function renderVisualReviewGallery(report) {
  const views = Array.isArray(report?.reviewed_views) ? report.reviewed_views : [];
  if (!views.length) return;
  const gallery = document.createElement("div");
  gallery.className = "visual-review-gallery";
  for (const view of views) {
    const card = document.createElement("section");
    card.className = "visual-review-view";
    const head = document.createElement("div");
    head.className = "visual-review-view-head";
    appendText(head, "strong", "", view.label || view.id || "View");
    appendText(head, "span", "", view.id || "");
    card.appendChild(head);
    const images = document.createElement("div");
    images.className = `visual-review-images ${view.images?.before ? "paired" : "single"}`;
    for (const phase of ["before", "after"]) {
      const image = view.images?.[phase];
      if (!image?.url) continue;
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = image.url;
      img.alt = `${phase} ${view.label || view.id} visual review evidence`;
      img.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = phase;
      figure.append(img, caption);
      images.appendChild(figure);
    }
    card.appendChild(images);
    gallery.appendChild(card);
  }
  els.visualReviewBody.appendChild(gallery);
}

function renderVisualReviewChecks(report) {
  const checks = Array.isArray(report?.checks) ? report.checks : [];
  if (!checks.length) return;
  const list = document.createElement("div");
  list.className = "visual-review-checks";
  for (const check of checks) {
    const item = document.createElement("div");
    const passed = check.passed === true;
    item.className = `visual-review-check ${passed ? "passed" : "failed"} ${check.severity || "advisory"}`;
    const top = document.createElement("div");
    top.className = "env-check-top";
    const title = document.createElement("div");
    title.className = "env-check-title";
    const indicator = document.createElement("span");
    indicator.className = `check-indicator ${passed ? "passed" : "failed"}`;
    const name = document.createElement("strong");
    name.textContent = check.message || check.id || "Visual check";
    title.append(indicator, name);
    const pill = document.createElement("span");
    pill.className = "env-check-pill";
    pill.textContent = passed ? "passed" : check.severity || "failed";
    top.append(title, pill);
    item.appendChild(top);
    const evidence = (check.evidence || []).map((entry) => {
      return `${entry.phase || "after"} ${entry.view_id || "view"}: ${entry.observation || ""}`;
    }).filter(Boolean);
    if (evidence.length) appendText(item, "div", "visual-review-evidence", evidence.join(" "));
    if (!passed && check.repair_hint) appendText(item, "div", "env-check-hint", check.repair_hint);
    list.appendChild(item);
  }
  els.visualReviewBody.appendChild(list);
}

function renderVisualReviewDetails(report) {
  if (!report?.prompt && !report?.raw_output) return;
  const details = document.createElement("details");
  details.className = "visual-review-details";
  const summary = document.createElement("summary");
  summary.textContent = "Review context";
  const pre = document.createElement("pre");
  pre.textContent = productDisplayText(
    [report.prompt, report.raw_output ? `\nReviewer output:\n${report.raw_output}` : ""].filter(Boolean).join("\n"),
  );
  details.append(summary, pre);
  els.visualReviewBody.appendChild(details);
}

function updateVisualTabBadge(summary, report) {
  const status = String(summary?.status || "not_run");
  const reportSummary = report?.summary || {};
  const total = Number(reportSummary.total || 0);
  const passed = Number(reportSummary.passed || 0);
  const critical = Number(reportSummary.critical_failures || 0);
  const advisory = Number(reportSummary.advisory_failures || 0);
  const running = ["pending_generation", "awaiting_capture", "evidence_ready", "reviewing"].includes(status);
  if (running) {
    renderProgressDots(els.visualTabBadge, "Visual review running");
  } else {
    els.visualTabBadge.removeAttribute("aria-label");
    els.visualTabBadge.textContent = total ? `${passed}/${total}` : status === "error" ? "!" : "0";
  }
  const badgeClass = running
    ? "live"
    : ["needs_attention", "error"].includes(status) || critical
      ? "bad"
      : status === "stale" || advisory
        ? "warn"
        : status === "passed"
          ? "good"
          : "";
  els.visualTabBadge.className = `tab-count ${badgeClass}`.trim();
}

function renderTasks(scene) {
  const tasks = Array.isArray(scene?.tasks) ? scene.tasks : [];
  const summary = scene?.task_summary || { total: tasks.length, validated: 0, stale: 0, label: "No tasks" };
  const canCreate = Boolean(scene?.status === "finalized" && scene?.capabilities?.task_testable);
  const taskRunning = Boolean(state.tasks.runningTaskId);
  els.taskSummary.textContent = taskRunning
    ? "Agent running"
    : summary.label || (tasks.length ? `${tasks.length} tasks` : "No tasks yet");
  if (taskRunning) {
    renderProgressDots(els.taskTabBadge, "Task agent running");
  } else {
    els.taskTabBadge.removeAttribute("aria-label");
    els.taskTabBadge.textContent = tasks.length ? `${Number(summary.validated || 0)}/${tasks.length}` : "0";
  }
  els.taskTabBadge.className = `tab-count ${taskRunning ? "live" : summary.stale ? "warn" : tasks.length && Number(summary.validated || 0) === tasks.length ? "good" : ""}`.trim();
  els.newTaskButton.disabled = !canCreate || state.running || state.tasks.compiling || state.tasks.oracle.active || Boolean(state.tasks.runningTaskId);
  els.newTaskButton.textContent = state.tasks.compiling ? "Generating..." : "New Task";
  els.taskCreateForm.classList.toggle("hidden", !state.tasks.composerOpen);
  els.taskInstruction.disabled = state.tasks.compiling;
  els.taskCreateCancel.disabled = state.tasks.compiling;
  els.taskCreateSubmit.disabled = state.tasks.compiling || !canCreate;
  renderTaskNotice();
  els.taskList.replaceChildren();

  if (!scene) return;
  if (!scene.capabilities?.task_testable) {
    appendTaskEmpty("No controllable agent", "Add a robot to this environment before defining trajectory tasks.");
    return;
  }
  if (!tasks.length) {
    appendTaskEmpty("No tasks yet", "Describe an objective, review the generated trajectory tests, then demonstrate a passing oracle in the scene.");
    return;
  }
  for (const task of tasks) els.taskList.appendChild(taskCard(task));
}

function renderTaskNotice() {
  const notice = state.tasks.notice;
  els.taskNotice.classList.toggle("hidden", !notice);
  els.taskNotice.className = `task-notice ${notice?.type || ""} ${notice ? "" : "hidden"}`.trim();
  els.taskNotice.textContent = notice?.message || "";
}

function appendTaskEmpty(title, message) {
  const node = document.createElement("div");
  node.className = "task-empty";
  appendText(node, "strong", "", title);
  appendText(node, "span", "", message);
  els.taskList.appendChild(node);
}

function taskCard(task) {
  const card = document.createElement("article");
  const running = state.tasks.runningTaskId === task.task_id;
  card.className = `task-card ${running ? "running" : ""}`.trim();
  card.dataset.taskId = task.task_id || "";
  const status = String(task.effective_status || task.status || "error");
  const head = document.createElement("div");
  head.className = "task-card-head";
  const title = document.createElement("div");
  title.className = "task-card-title";
  appendText(title, "strong", "", task.instruction || "Untitled task");
  const headSide = document.createElement("div");
  headSide.className = "task-card-head-side";
  const statusNode = appendText(headSide, "span", `task-status ${running ? "running" : status}`, running ? "" : taskStatusLabel(status));
  if (running) renderProgressDots(statusNode, "Task agent running", "Running");
  statusNode.title = task.stale_reason || task.compiler_error || "";

  const actions = document.createElement("div");
  actions.className = "task-card-actions";
  const secondaryActions = document.createElement("div");
  secondaryActions.className = "task-secondary-actions";
  if (["pending_oracle", "validation_failed", "stale", "validated", "recording"].includes(status)) {
    secondaryActions.appendChild(taskActionButton(status === "validated" ? "Rerecord Oracle" : "Record Oracle", "record", status !== "validated"));
  }
  if (task.oracle?.trajectory_url) secondaryActions.appendChild(taskActionButton("Replay Oracle", "replay", false, task.oracle.trajectory_url));
  if (status === "error") secondaryActions.appendChild(taskActionButton("Retry Tests", "retry", true));
  secondaryActions.appendChild(taskActionButton("Delete", "delete", false, "", "ghost danger"));
  if (status === "validated") {
    actions.appendChild(taskAgentControls(task));
    actions.appendChild(taskActionMenu(secondaryActions, task));
  } else {
    actions.appendChild(secondaryActions);
  }
  headSide.appendChild(actions);
  head.append(title, headSide);
  card.appendChild(head);

  const resultSelection = selectedTaskResult(task.task_id);
  const resultView = buildTaskResultView(task, resultSelection);
  const rows = resultView.rows;
  const runs = taskTrajectoryRuns(task);
  const content = document.createElement("div");
  content.className = `task-card-content ${runs.length ? "" : "without-runs"}`.trim();
  const tests = document.createElement("section");
  tests.className = "task-tests";
  const listHead = document.createElement("div");
  listHead.className = "task-tests-head";
  appendText(listHead, "strong", "", "Tests");
  appendText(listHead, "span", "", resultView.summary);
  tests.appendChild(listHead);
  const list = document.createElement("div");
  list.className = "task-test-list";
  for (const item of rows) {
    const row = document.createElement("div");
    const resultClass = item.passed === true ? "pass" : item.passed === false ? "fail" : "";
    row.className = `task-condition ${resultClass}`.trim();
    appendText(row, "span", "task-condition-dot", "");
    appendText(row, "span", "", item.description);
    list.appendChild(row);
  }
  tests.appendChild(list);
  content.appendChild(tests);

  if (runs.length) content.appendChild(taskTrajectoryHistory(task, runs));
  card.appendChild(content);
  return card;
}

function taskActionMenu(items, task) {
  const menu = document.createElement("details");
  menu.className = "task-action-menu";
  const trigger = document.createElement("summary");
  trigger.className = "task-action-menu-trigger";
  trigger.textContent = "...";
  trigger.setAttribute("aria-label", `More actions for ${task.instruction || task.task_id}`);
  trigger.title = "Task actions";
  items.classList.add("task-action-menu-items");
  menu.append(trigger, items);
  return menu;
}

function taskAgentControls(task) {
  const controls = document.createElement("div");
  controls.className = "task-agent-controls";
  const select = document.createElement("select");
  select.className = "task-agent-model";
  select.dataset.taskRunModel = "";
  select.setAttribute("aria-label", `Model for agent run: ${task.instruction || task.task_id}`);
  select.title = "Choose the Codex model for this agent run";
  populateCodexModelOptions(select, taskRunModelSelection(task.task_id), "Default");
  select.disabled = state.codexModels.status === "loading" || Boolean(state.tasks.runningTaskId);
  controls.append(select, taskActionButton("Run Agent", "run", true));
  return controls;
}

function taskRunModelKey(taskId) {
  return `${state.currentScene?.env_id || ""}:${taskId}`;
}

function taskResultKey(taskId) {
  return `${state.currentScene?.env_id || ""}:${taskId}`;
}

function selectedTaskResult(taskId) {
  return state.tasks.selectedResults.get(taskResultKey(taskId)) || null;
}

function taskRunModelSelection(taskId) {
  const key = taskRunModelKey(taskId);
  return state.tasks.runModels.has(key)
    ? String(state.tasks.runModels.get(key) || "")
    : String(state.codexModels.selected || "");
}

function resolvedTaskRunModel(selection) {
  return String(selection || state.codexModels.defaultModel?.id || "");
}

function taskActionButton(label, action, primary, url = "", extraClass = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${primary ? "primary" : "ghost"} ${extraClass}`.trim();
  button.dataset.taskAction = action;
  if (url) button.dataset.trajectoryUrl = url;
  button.textContent = label;
  button.disabled = state.tasks.compiling || state.tasks.oracle.active || Boolean(state.tasks.runningTaskId);
  return button;
}

function taskTrajectoryHistory(task, runs = taskTrajectoryRuns(task)) {
  const section = document.createElement("details");
  section.className = "task-trajectory-history";
  const expansionKey = taskRunModelKey(task.task_id);
  section.open = state.tasks.expandedRuns.has(expansionKey);
  section.addEventListener("toggle", () => {
    if (section.open) state.tasks.expandedRuns.add(expansionKey);
    else state.tasks.expandedRuns.delete(expansionKey);
  });
  const head = document.createElement("summary");
  head.className = "task-trajectory-head";
  appendText(head, "strong", "", "Agent runs");
  appendText(head, "span", "", `${runs.length} saved`);
  section.appendChild(head);

  const list = document.createElement("div");
  list.className = "task-trajectory-list";
  const selected = selectedTaskResult(task.task_id);
  for (const run of runs) {
    const row = document.createElement("div");
    const isSelected = selected?.kind === "run" && selected.id === run.run_id;
    row.className = `task-trajectory-row ${run.passed ? "pass" : run.status === "error" ? "error" : "fail"} ${isSelected ? "selected" : ""}`.trim();
    const identity = document.createElement("button");
    identity.type = "button";
    identity.className = "task-trajectory-identity";
    identity.dataset.taskAction = "select-trajectory";
    identity.dataset.runId = run.run_id;
    identity.setAttribute("aria-pressed", String(isSelected));
    identity.setAttribute("aria-label", `${isSelected ? "Clear" : "View"} results for run ${Number(run.run_number || 1)}`);
    const title = document.createElement("div");
    appendText(title, "span", "task-trajectory-dot", "");
    appendText(title, "strong", "", `Run ${Number(run.run_number || 1)}`);
    appendText(title, "span", "task-trajectory-status", taskTrajectoryStatus(run));
    identity.appendChild(title);
    const metadata = document.createElement("div");
    metadata.className = "task-trajectory-meta";
    appendText(metadata, "span", "", `${Number(run.step_count || 0)} steps`);
    appendText(metadata, "span", "", `${Number(run.action_count || 0)} actions`);
    appendText(metadata, "span", "", !run.model || run.model === "default" ? "Default model" : run.model);
    const completed = formatTaskRunTime(run.completed_at || run.created_at);
    if (completed) appendText(metadata, "span", "", completed);
    identity.appendChild(metadata);
    const replayButton = taskActionButton("Replay", "replay-trajectory", false, "", "task-replay-button");
    replayButton.dataset.runId = run.run_id;
    replayButton.setAttribute("aria-label", `Replay run ${Number(run.run_number || 1)} for ${task.instruction || task.task_id}`);
    row.append(identity, replayButton);
    list.appendChild(row);
  }
  section.appendChild(list);
  return section;
}

function taskTrajectoryStatus(run) {
  if (run.stale) return "Stale";
  if (run.passed) return "Passed";
  if (run.status === "error") return "Error";
  return "Did not pass";
}

function formatTaskRunTime(value) {
  const time = Date.parse(String(value || ""));
  if (!Number.isFinite(time)) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(time));
}

function taskRunPhaseLabel(status) {
  return {
    starting: "Starting",
    running: "Running",
    replaying: "Verifying replay",
    passed: "Passed",
    failed: "Did not pass",
    error: "Error",
  }[String(status || "")] || "Activity";
}

function taskActivityLabel(event) {
  if (event.type === "agent_message") return "Agent note";
  if (event.type === "error") return "Error";
  const names = {
    start_task_run: "Start",
    observe_task_run: "Observe",
    act_task_run: "Control",
    reset_task_run: "Reset",
    stop_task_run: "Finish",
  };
  return names[event.name] || event.label || "Activity";
}

function taskStatusLabel(status) {
  return {
    compiling: "Generating tests",
    pending_oracle: "Needs oracle",
    recording: "Recording",
    validation_failed: "Oracle failed",
    validated: "Validated",
    stale: "Stale",
    error: "Error",
  }[status] || status.replaceAll("_", " ");
}

function openTaskComposer() {
  if (state.tasks.compiling || state.tasks.oracle.active) return;
  state.tasks.composerOpen = true;
  state.tasks.notice = null;
  renderTasks(state.currentScene);
  queueMicrotask(() => els.taskInstruction.focus());
}

function closeTaskComposer() {
  if (state.tasks.compiling) return;
  state.tasks.composerOpen = false;
  state.tasks.notice = null;
  els.taskInstruction.value = "";
  renderTasks(state.currentScene);
}

async function submitTaskDefinition(event) {
  event.preventDefault();
  const instruction = els.taskInstruction.value.trim();
  if (!instruction || state.tasks.compiling || !state.currentScene?.env_id) return;
  await compileTaskInstruction(instruction);
}

async function compileTaskInstruction(instruction) {
  const envId = state.currentScene?.env_id;
  if (!envId) return;
  state.tasks.compiling = true;
  state.tasks.notice = { type: "running", message: "Codex is writing typed trajectory tests from this objective..." };
  renderEnvHeader(state.currentScene);
  renderTasks(state.currentScene);
  setStatus("Generating task tests", "running");
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(envId)}/tasks/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(withCodexModel({ instruction })),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Task request failed: ${response.status}`);
    if (data.scene) {
      state.currentScene = data.scene;
      mergeScene(data.scene);
    }
    if (data.status === "error") {
      state.tasks.notice = { type: "error", message: data.error || "Codex could not generate valid task tests." };
    } else {
      state.tasks.composerOpen = false;
      els.taskInstruction.value = "";
      state.tasks.notice = { message: "Tests generated. Review them below, then record the oracle solution." };
    }
  } catch (error) {
    state.tasks.notice = { type: "error", message: error.message || String(error) };
  } finally {
    state.tasks.compiling = false;
    render();
    setInspectorTab("tasks");
  }
}

async function handleTaskAction(event) {
  const button = event.target.closest("[data-task-action]");
  if (!button || button.disabled) return;
  button.closest(".task-action-menu")?.removeAttribute("open");
  const card = button.closest("[data-task-id]");
  const taskId = card?.dataset.taskId || "";
  const task = (state.currentScene?.tasks || []).find((item) => item.task_id === taskId);
  if (!task) return;
  const action = button.dataset.taskAction;
  if (action === "record") await startTaskOracle(task);
  else if (action === "replay" && button.dataset.trajectoryUrl) {
    if (task.last_validation) {
      state.tasks.selectedResults.set(taskResultKey(task.task_id), {
        kind: "oracle",
        id: String(task.oracle?.attempt_id || "latest"),
        label: "Oracle",
        report: task.last_validation,
        loading: false,
        error: "",
      });
      renderTasks(state.currentScene);
    }
    await startBehaviorReplay(button.dataset.trajectoryUrl, `oracle · ${taskId}`);
  }
  else if (action === "select-trajectory" && button.dataset.runId) {
    const run = taskTrajectoryRuns(task).find((item) => item.run_id === button.dataset.runId);
    if (run) await selectTaskTrajectory(task, run, { toggle: true });
  }
  else if (action === "replay-trajectory" && button.dataset.runId) {
    const run = taskTrajectoryRuns(task).find((item) => item.run_id === button.dataset.runId);
    if (run) {
      await selectTaskTrajectory(task, run, { toggle: false });
      await startTaskTrajectoryReplay(task, run);
    }
  }
  else if (action === "retry") {
    await deleteTask(task, { confirmDelete: false });
    await compileTaskInstruction(task.instruction || "");
  } else if (action === "delete") await deleteTask(task);
  else if (action === "run") {
    const select = card.querySelector("[data-task-run-model]");
    const selection = String(select?.value ?? taskRunModelSelection(task.task_id));
    state.tasks.runModels.set(taskRunModelKey(task.task_id), selection);
    await runValidatedTask(task, selection);
  }
}

async function selectTaskTrajectory(task, run, { toggle = true } = {}) {
  const key = taskResultKey(task.task_id);
  const current = state.tasks.selectedResults.get(key);
  if (toggle && current?.kind === "run" && current.id === run.run_id) {
    state.tasks.selectedResults.delete(key);
    renderTasks(state.currentScene);
    return;
  }

  const selection = {
    kind: "run",
    id: run.run_id,
    label: `Run ${Number(run.run_number || 1)}`,
    report: null,
    loading: true,
    error: "",
  };
  state.tasks.selectedResults.set(key, selection);
  renderTasks(state.currentScene);
  try {
    selection.report = await loadTaskResultReport(run.report_url);
    selection.loading = false;
  } catch (error) {
    selection.loading = false;
    selection.error = error.message || String(error);
  }
  if (state.tasks.selectedResults.get(key) === selection) renderTasks(state.currentScene);
}

async function loadTaskResultReport(url) {
  if (!url) throw new Error("This saved run has no result report.");
  const cached = state.tasks.resultReports.get(url);
  if (cached) return await cached;
  const request = fetch(url, { cache: "no-store" }).then(async (response) => {
    if (!response.ok) throw new Error(`Run report request failed: ${response.status}`);
    const report = await response.json();
    if (!report || !Array.isArray(report.tests)) throw new Error("Run report is malformed.");
    return report;
  });
  state.tasks.resultReports.set(url, request);
  try {
    const report = await request;
    state.tasks.resultReports.set(url, report);
    return report;
  } catch (error) {
    state.tasks.resultReports.delete(url);
    throw error;
  }
}

function closeOpenTaskActionMenus(event) {
  for (const menu of document.querySelectorAll(".task-action-menu[open]")) {
    if (!menu.contains(event.target)) menu.removeAttribute("open");
  }
}

function handleTaskRunModelChange(event) {
  const select = event.target.closest("[data-task-run-model]");
  if (!select) return;
  const taskId = select.closest("[data-task-id]")?.dataset.taskId || "";
  if (!taskId) return;
  state.tasks.runModels.set(taskRunModelKey(taskId), String(select.value || ""));
}

async function deleteTask(task, { confirmDelete = true } = {}) {
  if (confirmDelete && !window.confirm(`Delete task “${task.instruction || task.task_id}”?`)) return;
  const envId = state.currentScene?.env_id;
  if (!envId) return;
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(envId)}/tasks/${encodeURIComponent(task.task_id)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Delete failed: ${response.status}`);
    state.tasks.runModels.delete(taskRunModelKey(task.task_id));
    state.tasks.expandedRuns.delete(taskRunModelKey(task.task_id));
    state.tasks.selectedResults.delete(taskResultKey(task.task_id));
    const latest = await fetchScene(envId);
    state.currentScene = latest;
    mergeScene(latest);
    state.tasks.notice = null;
    render();
    setInspectorTab("tasks");
  } catch (error) {
    state.tasks.notice = { type: "error", message: error.message || String(error) };
    renderTasks(state.currentScene);
  }
}

async function startTaskOracle(task) {
  if (state.tasks.oracle.active || state.tasks.compiling || !state.currentScene) return;
  if (state.play.active) await stopPlayableEnvironment({ silent: true });
  if (state.behavior.replay.active) exitBehaviorReplay();
  state.tasks.selectedResults.delete(taskResultKey(task.task_id));
  renderTasks(state.currentScene);
  setStatus("Starting oracle", "running");
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(state.currentScene.env_id)}/tasks/${encodeURIComponent(task.task_id)}/oracle/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Oracle request failed: ${response.status}`);
    const oracle = state.tasks.oracle;
    Object.assign(oracle, {
      active: true,
      sessionId: data.state.session_id,
      taskId: task.task_id,
      instruction: task.instruction || "",
      agentId: data.state.agent_id || "",
      terminalReason: "",
      report: data.state.report || null,
      requestInFlight: false,
      hasStarted: false,
      hasActions: Boolean(data.state.has_actions),
      idleTicks: 0,
      readyToValidate: false,
      reportCurrent: true,
      lastTickAt: 0,
      simulationTimestep: Number(data.state.timestep || 0.01),
      finishPending: false,
      finishing: false,
    });
    const visibleTask = (state.currentScene?.tasks || []).find((item) => item.task_id === task.task_id);
    if (visibleTask) {
      visibleTask.status = "recording";
      visibleTask.effective_status = "recording";
    }
    oracle.keys.clear();
    oracle.keyPulseUntil.clear();
    els.taskOracleControls.classList.remove("hidden");
    els.taskOracleInstruction.textContent = oracle.instruction;
    activateAgentCamera({ agentId: oracle.agentId });
    applyTaskOracleState(data.state);
    updatePlayModeControls();
    updatePlayButton(state.currentScene);
    renderTasks(state.currentScene);
    setStatus("Oracle ready for input", "ready");
  } catch (error) {
    state.tasks.notice = { type: "error", message: `Oracle could not start: ${error.message || String(error)}` };
    renderTasks(state.currentScene);
    setStatus("Oracle failed", "error");
  }
}

function scheduleTaskOracleTick(delay = PLAY_TICK_MS) {
  const oracle = state.tasks.oracle;
  if (!oracle.active || oracle.terminalReason || oracle.readyToValidate || oracle.finishPending || oracle.finishing) return;
  if (oracle.timer !== null) window.clearTimeout(oracle.timer);
  oracle.timer = window.setTimeout(runTaskOracleTick, delay);
}

async function runTaskOracleTick() {
  const oracle = state.tasks.oracle;
  if (!oracle.active || !oracle.sessionId || oracle.terminalReason || oracle.finishPending || oracle.finishing) return;
  if (oracle.requestInFlight) {
    scheduleTaskOracleTick();
    return;
  }
  const sessionId = oracle.sessionId;
  const input = currentControllerInput(oracle);
  const activeInput = Boolean(input.right || input.forward || input.jump);
  if (activeInput) {
    oracle.hasStarted = true;
    oracle.idleTicks = 0;
  } else if (oracle.hasStarted) {
    oracle.idleTicks += 1;
  }
  if (!oracle.hasStarted || oracle.idleTicks > ORACLE_IDLE_TICKS_AFTER_INPUT) {
    oracle.timer = null;
    oracle.lastTickAt = 0;
    if (!oracle.readyToValidate && !oracle.terminalReason) setStatus("Oracle paused", "ready");
    return;
  }
  const now = performance.now();
  const frames = recordingFrameCount({
    elapsedMs: oracle.lastTickAt ? now - oracle.lastTickAt : PLAY_TICK_MS,
    timestep: oracle.simulationTimestep,
  });
  const evaluateReport = recordingReportRequested({
    activeInput,
    idleTicks: oracle.idleTicks,
    idleLimit: ORACLE_IDLE_TICKS_AFTER_INPUT,
  });
  oracle.lastTickAt = now;
  oracle.requestInFlight = true;
  updateTaskOracleFinishButton();
  try {
    const response = await fetch(`/api/task-oracle/${encodeURIComponent(sessionId)}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...input,
        camera_azimuth: state.preview.visual.azimuth,
        frames,
        evaluate_report: evaluateReport,
      }),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Oracle step failed: ${response.status}`);
    if (oracle.active && oracle.sessionId === sessionId) applyTaskOracleState(data.state);
  } catch (error) {
    state.tasks.notice = { type: "error", message: `Oracle recording stopped: ${error.message || String(error)}` };
    await cancelTaskOracle({ silent: true });
    renderTasks(state.currentScene);
  } finally {
    oracle.requestInFlight = false;
    updateTaskOracleFinishButton();
    if (oracle.active && oracle.finishPending) void completeTaskOracleFinish();
    else if (oracle.active && !oracle.terminalReason && !oracle.readyToValidate) scheduleTaskOracleTick();
  }
}

function applyTaskOracleState(oracleState) {
  if (!oracleState) return;
  const oracle = state.tasks.oracle;
  oracle.agentId = oracleState.agent_id || oracle.agentId;
  oracle.terminalReason = oracleState.terminal_reason || "";
  oracle.report = oracleState.report || oracle.report;
  oracle.readyToValidate = Boolean(oracleState.ready_to_validate);
  oracle.reportCurrent = oracleState.report_current !== false;
  oracle.hasActions = Boolean(oracleState.has_actions);
  oracle.simulationTimestep = Number(oracleState.timestep || oracle.simulationTimestep || 0.01);
  state.preview.renderer?.applyPhysicsState(oracleState.objects || []);
  state.preview.renderer?.applyGameState?.({ state: "playing", mechanisms: oracleState.mechanisms || [] });
  const summary = oracle.report?.summary || {};
  els.taskOracleChecks.textContent = oracle.reportCurrent
    ? `${Number(summary.passed_conditions || 0)} of ${Number(summary.conditions || 0)} tests passed`
    : "Tests update when movement pauses";
  updateTaskOracleFinishButton();
  els.taskOracleChecklist.replaceChildren();
  for (const test of oracle.report?.tests || []) {
    for (const condition of test.conditions || []) {
      appendText(
        els.taskOracleChecklist,
        "span",
        `task-oracle-check ${oracle.reportCurrent && condition.passed ? "pass" : ""}`,
        condition.description || condition.id || "Check",
      );
    }
  }
  els.previewStage.dataset.agentPosition = JSON.stringify((oracleState.objects || []).find((item) => item.id === oracle.agentId)?.position || []);
  els.previewStage.dataset.grounded = String(Boolean(oracleState.grounded));
  if (oracle.terminalReason) {
    setStatus(oracle.terminalReason === "step_budget" ? "Step budget reached" : "Attempt failed", "error");
    state.tasks.notice = {
      type: "error",
      message: oracle.terminalReason === "step_budget"
        ? "The attempt reached its step budget. Reset to try again or validate this failed attempt."
        : `The attempt ended on ${oracle.terminalReason.replaceAll("_", " ")}. Reset to try again or validate the evidence.`,
    };
    renderTaskNotice();
  } else if (oracleState.ready_to_validate) {
    setStatus("Oracle passes live checks", "done");
  }
}

async function resetTaskOracle() {
  const oracle = state.tasks.oracle;
  if (!oracle.active || !oracle.sessionId) return;
  oracle.keys.clear();
  oracle.keyPulseUntil.clear();
  if (oracle.timer !== null) window.clearTimeout(oracle.timer);
  oracle.timer = null;
  try {
    const response = await fetch(`/api/task-oracle/${encodeURIComponent(oracle.sessionId)}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Oracle reset failed: ${response.status}`);
    oracle.terminalReason = "";
    oracle.hasStarted = false;
    oracle.hasActions = false;
    oracle.idleTicks = 0;
    oracle.readyToValidate = false;
    oracle.reportCurrent = true;
    oracle.lastTickAt = 0;
    oracle.finishPending = false;
    oracle.finishing = false;
    state.tasks.notice = null;
    applyTaskOracleState(data.state);
    renderTaskNotice();
    setStatus("Oracle ready for input", "ready");
  } catch (error) {
    state.tasks.notice = { type: "error", message: error.message || String(error) };
    renderTaskNotice();
  }
}

async function finishTaskOracle() {
  const oracle = state.tasks.oracle;
  const intent = recordingFinishIntent(oracle);
  if (!intent.accepted) return;
  if (oracle.timer !== null) window.clearTimeout(oracle.timer);
  oracle.timer = null;
  oracle.keys.clear();
  oracle.keyPulseUntil.clear();
  oracle.finishPending = true;
  updateTaskOracleFinishButton();
  if (!intent.startImmediately) return;
  await completeTaskOracleFinish();
}

async function completeTaskOracleFinish() {
  const oracle = state.tasks.oracle;
  if (!oracle.active || !oracle.sessionId || !oracle.finishPending || oracle.finishing || oracle.requestInFlight) return;
  const completedTaskId = oracle.taskId;
  oracle.finishPending = false;
  oracle.finishing = true;
  oracle.requestInFlight = true;
  updateTaskOracleFinishButton();
  setStatus("Validating replay", "running");
  try {
    const response = await fetch(`/api/task-oracle/${encodeURIComponent(oracle.sessionId)}/finish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Oracle validation failed: ${response.status}`);
    clearTaskOracleState();
    if (data.scene) {
      state.currentScene = data.scene;
      mergeScene(data.scene);
    } else if (state.currentScene?.env_id) {
      state.currentScene = await fetchScene(state.currentScene.env_id);
      mergeScene(state.currentScene);
    }
    if (data.report) {
      state.tasks.selectedResults.set(taskResultKey(completedTaskId), {
        kind: "oracle",
        id: String(data.attempt?.attempt_id || "latest"),
        label: "Oracle",
        report: data.report,
        loading: false,
        error: "",
      });
    }
    state.tasks.notice = data.passed
      ? { message: "Oracle replay passed every test. The task is validated and ready for benchmark runs." }
      : { type: "error", message: "The authoritative replay failed one or more tests. Review the results and record another attempt." };
    render();
    setInspectorTab("tasks");
    setStatus(data.passed ? "Task validated" : "Oracle failed", data.passed ? "done" : "error");
  } catch (error) {
    oracle.requestInFlight = false;
    oracle.finishing = false;
    state.tasks.notice = { type: "error", message: error.message || String(error) };
    renderTaskNotice();
    updateTaskOracleFinishButton();
    scheduleTaskOracleTick();
    setStatus("Validation failed", "error");
  }
}

function updateTaskOracleFinishButton() {
  const view = recordingFinishView(state.tasks.oracle);
  els.taskOracleFinish.disabled = view.disabled;
  els.taskOracleFinish.textContent = view.label;
}

async function cancelTaskOracle({ silent = false, resetPreview = true } = {}) {
  const sessionId = state.tasks.oracle.sessionId;
  if (state.tasks.oracle.timer !== null) window.clearTimeout(state.tasks.oracle.timer);
  clearTaskOracleState();
  if (sessionId) {
    try {
      await fetch(`/api/task-oracle/${encodeURIComponent(sessionId)}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        keepalive: true,
      });
    } catch {
      // The server expires abandoned sessions; cancellation is best effort on navigation.
    }
  }
  if (state.currentScene?.env_id) {
    try {
      const latest = await fetchScene(state.currentScene.env_id);
      state.currentScene = latest;
      mergeScene(latest);
    } catch {
      // Keep the last scene snapshot if refresh fails.
    }
  }
  if (resetPreview && state.route.page === "env") render();
  if (!silent) setStatus("Oracle cancelled", "ready");
}

function clearTaskOracleState() {
  const oracle = state.tasks.oracle;
  if (oracle.timer !== null) window.clearTimeout(oracle.timer);
  oracle.keys.clear();
  oracle.keyPulseUntil.clear();
  Object.assign(oracle, {
    active: false,
    sessionId: "",
    taskId: "",
    instruction: "",
    agentId: "",
    timer: null,
    requestInFlight: false,
    terminalReason: "",
    report: null,
    hasStarted: false,
    hasActions: false,
    idleTicks: 0,
    readyToValidate: false,
    reportCurrent: true,
    lastTickAt: 0,
    simulationTimestep: 0.01,
    finishPending: false,
    finishing: false,
  });
  els.taskOracleControls.classList.add("hidden");
  els.taskOracleChecklist.replaceChildren();
  updatePlayModeControls();
  updatePlayButton(state.currentScene);
}

function stopTaskOracleOnPageExit() {
  const sessionId = state.tasks.oracle.sessionId;
  if (!sessionId) return;
  const body = new Blob(["{}"], { type: "application/json" });
  navigator.sendBeacon(`/api/task-oracle/${encodeURIComponent(sessionId)}/cancel`, body);
}

async function runValidatedTask(task, modelSelection = taskRunModelSelection(task.task_id)) {
  if (state.tasks.runningTaskId || state.tasks.oracle.active) return;
  const envId = state.currentScene?.env_id;
  if (!envId) return;
  const runState = {
    envId,
    taskId: task.task_id,
    runId: "",
    status: "starting",
    instruction: task.instruction || "",
    events: [],
    latestFrame: null,
    latestSceneFrame: null,
    sceneQueue: [],
    sceneTimer: null,
    donePayload: null,
    renderPending: false,
  };
  state.tasks.runningTaskId = task.task_id;
  state.tasks.activeRun = runState;
  state.tasks.notice = null;
  renderTasks(state.currentScene);
  // The agent-run endpoint is intentionally separate from oracle validation.
  try {
    await startTaskLivePreview(runState);
    const response = await fetch(`/api/scenes/${encodeURIComponent(envId)}/tasks/${encodeURIComponent(task.task_id)}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ model: resolvedTaskRunModel(modelSelection), stream: true }),
    });
    if (!response.ok) throw new Error(`Task run failed: ${response.status}`);
    const contentType = response.headers.get("Content-Type") || "";
    const data = contentType.includes("text/event-stream")
      ? await consumeTaskRunStream(response, runState)
      : await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Task run failed: ${response.status}`);
    if (data.scene) {
      state.currentScene = data.scene;
      mergeScene(data.scene);
    }
    runState.status = data.report?.passed ? "passed" : "failed";
    updateTaskRunOverlay(runState);
    state.tasks.notice = data.report?.passed
      ? { message: "The agent passed the task. The result was verified by an authoritative replay." }
      : { type: "warn", message: "The agent finished, but its replay did not pass every deterministic test." };
  } catch (error) {
    appendTaskRunEvent(runState, {
      type: "error",
      label: "Error",
      message: error.message || String(error),
      isError: true,
    });
    runState.status = "error";
    state.tasks.notice = { type: "error", message: error.message || String(error) };
  } finally {
    await stopTaskLivePreview(runState, { hold: runState.status === "error" ? 0 : 650 });
    state.tasks.runningTaskId = "";
    state.tasks.activeRun = null;
    render();
    setInspectorTab("tasks");
  }
}

async function consumeTaskRunStream(response, runState) {
  await consumeEventStream(response, {
    task_run(payload) {
      runState.status = String(payload.status || runState.status);
      runState.runId = String(payload.run_id || payload.manifest?.run_id || payload.report?.run_id || runState.runId);
      refreshTaskRunUi(runState);
    },
    text(payload) {
      const message = String(payload.delta || "").trim();
      if (message) appendTaskRunEvent(runState, { type: "agent_message", label: "Agent note", message, step: payload.step });
    },
    tool_start(payload) {
      appendTaskRunEvent(runState, {
        type: "tool_start",
        id: payload.id || "",
        name: payload.displayName || payload.name || "",
        label: "Action",
        message: payload.message || `Using ${payload.displayName || payload.name || "task tool"}...`,
        step: payload.step,
      });
    },
    tool_result(payload) {
      appendTaskRunEvent(runState, {
        type: "tool_result",
        id: payload.toolUseId || "",
        name: payload.displayName || payload.name || "",
        label: "Result",
        message: payload.message || `${payload.displayName || payload.name || "Task tool"} finished.`,
        isError: Boolean(payload.isError),
        summary: payload.summary || {},
        step: payload.step ?? payload.summary?.step,
      });
    },
    task_frame(payload) {
      if (!payload.url) return;
      runState.latestFrame = {
        type: "frame",
        url: payload.url,
        step: payload.step,
        reset_count: payload.reset_count,
      };
      refreshTaskRunUi(runState);
    },
    task_scene_frame(payload) {
      enqueueTaskSceneFrame(runState, payload);
    },
    task_error(payload) {
      appendTaskRunEvent(runState, {
        type: "error",
        label: "Agent error",
        message: payload.message || "Agent run failed.",
        isError: true,
        step: payload.step,
      });
    },
    stderr(payload) {
      console.debug("task agent stderr", payload.line);
    },
    done(payload) {
      runState.donePayload = payload;
      if (payload.isError) runState.status = "error";
    },
  });
  return runState.donePayload || { error: "Task stream ended without a final result." };
}

function appendTaskRunEvent(runState, event) {
  runState.events.push(event);
  if (runState.events.length > 80) runState.events = runState.events.slice(-80);
  refreshTaskRunUi(runState);
}

function refreshTaskRunUi(runState) {
  if (state.currentScene?.env_id !== runState.envId) return;
  updateTaskRunOverlay(runState);
  if (runState.renderPending) return;
  runState.renderPending = true;
  window.requestAnimationFrame(() => {
    runState.renderPending = false;
    if (state.currentScene?.env_id === runState.envId) renderTasks(state.currentScene);
  });
}

async function startTaskLivePreview(runState) {
  if (state.play.active) await stopPlayableEnvironment({ silent: true, resetPreview: false });
  if (state.behavior.replay.active) exitBehaviorReplay();
  if (state.currentScene?.env_id !== runState.envId) return;
  state.preview.visual = clonePreviewView(currentPreviewView());
  state.preview.mode = "visual";
  const ready = renderPreview();
  mountTaskRunOverlay(runState);
  els.previewStage.scrollIntoView({ behavior: "smooth", block: "center" });
  await ready;
  if (state.tasks.activeRun !== runState || state.currentScene?.env_id !== runState.envId) return;
  mountTaskRunOverlay(runState);
}

function mountTaskRunOverlay(runState) {
  if (state.currentScene?.env_id !== runState.envId) return;
  if (!els.taskRunOverlay.isConnected) els.previewStage.appendChild(els.taskRunOverlay);
  els.previewStage.classList.toggle("is-task-running", !runState.isReplay);
  els.previewStage.classList.toggle("is-task-replay", Boolean(runState.isReplay));
  els.taskRunOverlay.classList.remove("hidden");
  updateTaskRunOverlay(runState);
}

function enqueueTaskSceneFrame(runState, frame) {
  if (state.tasks.activeRun !== runState || !Array.isArray(frame?.objects)) return;
  runState.latestSceneFrame = frame;
  runState.sceneQueue.push(frame);
  if (runState.sceneQueue.length > 120) {
    runState.sceneQueue.splice(0, runState.sceneQueue.length - 60);
  }
  if (runState.sceneTimer === null) playNextTaskSceneFrame(runState);
}

function playNextTaskSceneFrame(runState) {
  if (state.tasks.activeRun !== runState || !runState.sceneQueue.length) {
    runState.sceneTimer = null;
    return;
  }
  const skip = runState.sceneQueue.length > 24
    ? Math.ceil(runState.sceneQueue.length / 12)
    : 1;
  const frame = runState.sceneQueue.splice(0, skip).at(-1);
  applyTaskSceneFrame(runState, frame);
  const delay = runState.sceneQueue.length > 18 ? 24 : 50;
  runState.sceneTimer = window.setTimeout(() => playNextTaskSceneFrame(runState), delay);
}

function applyTaskSceneFrame(runState, frame) {
  if (state.currentScene?.env_id !== runState.envId || !state.preview.renderer) return;
  state.preview.renderer.applyPhysicsState(frame.objects || []);
  state.preview.renderer.applyGameState?.({ mechanisms: frame.mechanisms || [] });
  els.previewStage.dataset.taskStep = String(Number(frame.step || 0));
  els.previewStage.dataset.grounded = String(Boolean(frame.grounded));
  updateTaskRunOverlay(runState);
}

function updateTaskRunOverlay(runState) {
  if (!els.taskRunOverlay.isConnected || state.currentScene?.env_id !== runState.envId) return;
  const active = !runState.isReplay && ["starting", "running", "replaying"].includes(String(runState.status || ""));
  els.taskRunOverlayTitle.textContent = runState.isReplay
    ? `Task trajectory${runState.runNumber ? ` · Run ${runState.runNumber}` : ""}`
    : "Task agent";
  els.taskRunOverlay.dataset.status = String(runState.status || "");
  els.taskRunOverlayStatus.className = `task-run-overlay-status ${active ? "running" : ""}`.trim();
  if (active) {
    renderProgressDots(
      els.taskRunOverlayStatus,
      "Task agent running",
      taskRunPhaseLabel(runState.status),
    );
  } else {
    els.taskRunOverlayStatus.replaceChildren();
    els.taskRunOverlayStatus.textContent = runState.isReplay
      ? (runState.playing ? "Replaying" : "Paused")
      : taskRunPhaseLabel(runState.status);
    els.taskRunOverlayStatus.removeAttribute("aria-label");
  }
  els.taskRunOverlayLog.replaceChildren();
  const events = runState.events || [];
  const latestNote = [...events].reverse().find((event) => event?.type === "agent_message");
  const recentActions = events
    .filter((event) => !["agent_message", "frame", "phase"].includes(String(event?.type || "")))
    .slice(-2);
  const visible = latestNote ? [latestNote, ...recentActions] : recentActions;
  if (!visible.length) {
    appendText(
      els.taskRunOverlayLog,
      "div",
      "task-run-overlay-row note",
      runState.instruction || "Starting the task agent...",
    );
  } else {
    for (const event of visible) {
      const row = document.createElement("div");
      row.className = `task-run-overlay-row ${event.isError || event.type === "error" ? "error" : ""} ${event.type === "agent_message" ? "note" : ""}`.trim();
      appendText(row, "strong", "", taskActivityLabel(event));
      appendText(row, "span", "", String(event.message || "").slice(0, 420));
      els.taskRunOverlayLog.appendChild(row);
    }
  }
  const frame = runState.latestSceneFrame;
  els.taskRunOverlayDetail.textContent = frame
    ? `Step ${Number(frame.step || 0)}${frame.grounded ? " · grounded" : ""}`
    : "Preparing the live scene";
}

function updateTaskTrajectoryReplayOverlay(replay, frame) {
  const overlay = replay.overlayState;
  if (!overlay) return;
  const step = Number(frame?.total_step ?? frame?.step ?? 0);
  overlay.events = taskActivityAtStep(replay.taskActivity, step);
  overlay.latestSceneFrame = {
    step,
    grounded: Boolean(frame?.grounded),
  };
  overlay.playing = replay.playing;
  els.previewStage.dataset.taskStep = String(step);
  els.previewStage.dataset.grounded = String(Boolean(frame?.grounded));
  updateTaskRunOverlay(overlay);
}

async function stopTaskLivePreview(runState, { hold = 0 } = {}) {
  if (runState.sceneTimer !== null) window.clearTimeout(runState.sceneTimer);
  runState.sceneTimer = null;
  if (runState.latestSceneFrame) applyTaskSceneFrame(runState, runState.latestSceneFrame);
  runState.sceneQueue = [];
  updateTaskRunOverlay(runState);
  if (hold > 0 && state.currentScene?.env_id === runState.envId && els.taskRunOverlay.isConnected) {
    await new Promise((resolve) => window.setTimeout(resolve, hold));
  }
  els.taskRunOverlay.classList.add("hidden");
  els.previewStage.classList.remove("is-task-running", "is-task-replay");
  delete els.previewStage.dataset.taskStep;
  if (!state.play.active && !state.tasks.oracle.active) delete els.previewStage.dataset.grounded;
}

function uniqueBehaviorTrialIds(values) {
  return [...new Set((values || []).map((value) => String(value || "")).filter(Boolean))];
}

function behaviorRequestsForEnv(envId) {
  if (!envId) return [];
  return [...state.behavior.requests.values()].filter((request) => request.envId === envId);
}

function serverActiveBehaviorRuns(summary) {
  if (Array.isArray(summary?.active_runs)) return summary.active_runs.filter(Boolean);
  return summary?.active_run ? [summary.active_run] : [];
}

function hasServerActiveBehaviorRuns(summary) {
  return serverActiveBehaviorRuns(summary).length > 0;
}

function behaviorActiveForScene(scene) {
  return Boolean(
    scene?.env_id
    && (behaviorRequestsForEnv(scene.env_id).length || hasServerActiveBehaviorRuns(scene.env_behavior_trials)),
  );
}

function renderBehaviorTrials(scene) {
  els.behaviorTrialStats.replaceChildren();
  els.behaviorTrialsBody.replaceChildren();
  const summary = scene?.env_behavior_trials || { status: "missing", label: "Behavior trials: not defined" };
  const plan = scene?.env_behavior_trial_plan;
  const persistedReport = scene?.env_behavior_trial_report;
  const trials = Array.isArray(plan?.trials) ? plan.trials : [];
  const localRequests = behaviorRequestsForEnv(scene?.env_id);
  const requestedTrialIds = uniqueBehaviorTrialIds(
    localRequests.flatMap((request) => request.trialIds || []),
  );
  const runView = buildBehaviorRunView({
    summary,
    report: persistedReport,
    trialIds: trials.map((trial) => trial.id),
    localRunning: localRequests.length > 0,
    requestedTrialIds,
  });
  const active = runView.active;
  const report = runView.displayReport;
  const activeTrialIds = new Set(runView.activeTrialIds);
  if (hasServerActiveBehaviorRuns(summary) && !localRequests.length) scheduleBehaviorTrialPoll(scene?.env_id);
  const status = String(summary.status || "missing");
  const invalidSetup = Number(report?.summary?.invalid_setup || 0) > 0;
  const headerView = buildBehaviorHeaderView({
    summary,
    report,
    trialCount: trials.length,
    active,
    runningCount: runView.runningCount,
  });
  els.behaviorTrialsSummary.textContent = headerView.label;
  for (const stat of headerView.stats) appendBehaviorStat(stat.label, stat.value, stat.tone);
  updateBehaviorTabBadge(summary, report, trials.length);
  els.behaviorRunAll.textContent = report ? "Rerun All Tests" : "Run All Tests";
  els.behaviorRunAll.classList.toggle("hidden", active || trials.length < 2 || invalidSetup || ["not_applicable", "stale", "dismissed"].includes(status));
  els.behaviorDismissAll.classList.toggle("hidden", active || !trials.length || ["dismissed", "stale"].includes(status));
  const behaviorActionStatus = String(report?.status || status);
  const shouldRegenerate = status === "stale" || invalidSetup;
  els.behaviorRepair.textContent = shouldRegenerate ? "Update Tests" : "Fix Environment";
  els.behaviorRepair.className = shouldRegenerate ? "primary compact" : "ghost";
  els.behaviorRepair.classList.toggle("hidden", active || !["failed", "needs_attention", "stale"].includes(status === "stale" ? status : behaviorActionStatus));
  els.behaviorRunAll.disabled = active;
  els.behaviorDismissAll.disabled = active;
  els.behaviorRepair.disabled = active || state.running;

  if (status === "not_applicable") {
    appendBehaviorNotice("No controllable agent", "Agent tests require a controllable robot in the environment.", "quiet");
    return;
  }
  if (!plan || !trials.length) {
    appendBehaviorNotice("No agent tests yet", "Codex will add tests for the agent behaviors that matter in this environment.", "quiet");
    return;
  }
  if (status === "stale") {
    appendBehaviorNotice("Agent tests need updating", "The environment changed. Update the tests before relying on the previous result.", "warn");
    if (persistedReport) renderStaleBehaviorHistory();
    return;
  }

  const resultById = new Map((runView.retainedReport?.results || []).map((item) => [String(item.trial_id), item]));
  const list = document.createElement("div");
  list.className = "behavior-trial-list";
  for (const [index, trial] of trials.entries()) {
    const trialActive = activeTrialIds.has(String(trial.id));
    const result = trialActive ? null : resultById.get(String(trial.id));
    const card = document.createElement("section");
    const resultStatus = String(trialActive ? "running" : result?.status || "ready");
    card.className = `behavior-trial-card ${resultStatus}`;
    const head = document.createElement("div");
    head.className = "behavior-trial-card-head";
    const identity = document.createElement("div");
    appendText(identity, "span", "behavior-trial-index", `Test ${index + 1}`);
    appendText(identity, "strong", "", trial.instruction || trial.id || "Agent test");
    const pill = document.createElement("span");
    pill.className = `behavior-trial-pill ${resultStatus}`;
    if (trialActive) renderProgressDots(pill, "Agent test running", "Running");
    else pill.textContent = behaviorResultLabel(resultStatus, trial.expected_outcome);
    head.append(identity, pill);
    card.appendChild(head);
    if (!trialActive && (!result || trial.expected_outcome === "should_not_succeed")) {
      appendText(
        card,
        "p",
        "behavior-trial-expectation",
        trial.expected_outcome === "should_not_succeed"
          ? "This test fails if the agent demonstrates the prohibited behavior."
          : "This test passes when the agent completes the behavior.",
      );
    }
    const negativeOutcome = result
      ? buildBehaviorOutcomeView({
          status: resultStatus,
          expectedOutcome: trial.expected_outcome,
        })
      : null;
    if (negativeOutcome) {
      renderBehaviorResultRow(card, negativeOutcome);
    } else {
      renderBehaviorObjective(
        card,
        result?.objective || trial.objective,
        result
          ? "Verified outcome"
          : trial.expected_outcome === "should_not_succeed"
            ? "Fails if"
            : "Passes when",
      );
    }
    const constraints = result?.constraints || trial.constraints;
    if (Array.isArray(constraints?.checks) && constraints.checks.length) {
      renderBehaviorObjective(card, constraints, "Rules");
    }
    if (result && resultStatus !== "passed") {
      if (!negativeOutcome) {
        appendText(card, "p", "behavior-outcome-summary", behaviorOutcomeSummary(result));
      }
    }
    const footer = document.createElement("div");
    footer.className = "behavior-run-footer";
    const meta = document.createElement("div");
    meta.className = "behavior-trial-meta";
    if (result) {
      appendText(meta, "span", "", `${Number(result.steps_used || 0)} steps`);
      appendText(meta, "span", "", `${Number(result.attempt_count || 0)} ${Number(result.attempt_count || 0) === 1 ? "attempt" : "attempts"}`);
      appendText(meta, "span", "", behaviorTerminationLabel(result.termination_reason));
    }
    if (meta.childElementCount) footer.appendChild(meta);
    if (!trialActive) {
      const actions = document.createElement("div");
      actions.className = "behavior-card-actions";
      const run = document.createElement("button");
      run.type = "button";
      run.className = "primary compact";
      run.dataset.behaviorAction = "run";
      run.dataset.trialId = trial.id;
      run.textContent = result ? "Rerun" : "Run Test";
      run.disabled = status === "stale" || invalidSetup;
      actions.appendChild(run);
      if (result?.trajectory_url) {
        const replay = document.createElement("button");
        replay.type = "button";
        replay.className = "ghost";
        replay.dataset.behaviorAction = "replay";
        replay.dataset.trialId = trial.id;
        replay.dataset.trajectoryUrl = result.trajectory_url;
        replay.textContent = "Replay in Scene";
        actions.appendChild(replay);
      }
      footer.appendChild(actions);
    }
    if (footer.childElementCount) card.appendChild(footer);
    if (result) {
      renderBehaviorEvidence(
        card,
        result.evidence_frames || [],
        result.trajectory_url || "",
        trial.id || "trial",
        scene?.visual_scene || null,
        scene?.mechanisms || [],
        result.child_summary || "",
        result.objective || null,
      );
    }
    list.appendChild(card);
  }
  els.behaviorTrialsBody.appendChild(list);
}

function scheduleBehaviorTrialPoll(envId) {
  if (!envId || state.behavior.pollTimer !== null) return;
  state.behavior.pollTimer = window.setTimeout(async () => {
    state.behavior.pollTimer = null;
    try {
      const latest = await fetchScene(envId);
      mergeScene(latest);
      if (state.currentScene?.env_id === envId) {
        state.currentScene = latest;
        renderEnvHeader(latest);
        renderBehaviorTrials(latest);
      }
      if (hasServerActiveBehaviorRuns(latest.env_behavior_trials)) scheduleBehaviorTrialPoll(envId);
    } catch {
      scheduleBehaviorTrialPoll(envId);
    }
  }, 1500);
}

function renderBehaviorObjective(container, objective, title = "Objective") {
  const checks = Array.isArray(objective?.checks) ? objective.checks : [];
  if (!checks.length) return;
  const head = document.createElement("div");
  head.className = "behavior-objective-head";
  appendText(head, "span", "behavior-objective-label", title);
  const completed = checks.filter((check) => check.passed === true).length;
  const evaluated = checks.filter((check) => typeof check.passed === "boolean").length;
  if (evaluated) appendText(head, "span", "behavior-objective-progress", `${completed}/${checks.length} verified`);
  container.appendChild(head);
  const list = document.createElement("div");
  list.className = "behavior-objective-list";
  for (const check of checks) {
    const row = document.createElement("div");
    const hasResult = typeof check.passed === "boolean";
    row.className = `behavior-objective-check ${hasResult ? check.passed ? "passed" : "failed" : "pending"}`;
    const marker = document.createElement("span");
    marker.className = "check-indicator";
    const label = document.createElement("span");
    label.textContent = describeBehaviorCheck(check);
    row.append(marker, label);
    const metric = hasResult && check.metrics ? formatBehaviorMetric(check.metrics, check) : "";
    if (metric && !(check.passed && metric === "1")) appendText(row, "code", "", metric);
    list.appendChild(row);
  }
  container.appendChild(list);
}

function renderBehaviorResultRow(container, outcome) {
  const head = document.createElement("div");
  head.className = "behavior-objective-head";
  appendText(head, "span", "behavior-objective-label", "Result");
  container.appendChild(head);
  const list = document.createElement("div");
  list.className = "behavior-objective-list";
  const row = document.createElement("div");
  row.className = `behavior-objective-check ${outcome.tone}`;
  const marker = document.createElement("span");
  marker.className = "check-indicator";
  const label = document.createElement("span");
  label.textContent = outcome.label;
  row.append(marker, label);
  list.appendChild(row);
  container.appendChild(list);
}

function renderBehaviorEvidence(
  container,
  frames,
  trajectoryUrl = "",
  trialId = "trial",
  visualScene = null,
  mechanisms = [],
  childSummary = "",
  objective = null,
) {
  const valid = (frames || [])
    .filter((frame) => frame?.url)
    .slice(0, 6)
    .map((frame) => ({ ...frame, label: describeBehaviorMilestone(frame, objective) }));
  const replayVerified = valid.every((frame) => frame.replay_verified === true);
  if (valid.length && trajectoryUrl && visualScene) {
    const section = document.createElement("section");
    section.className = "behavior-milestone-section";
    const header = document.createElement("div");
    header.className = "behavior-milestone-head";
    appendText(header, "strong", "", "Replay evidence");
    appendText(header, "span", "", `${valid.length} verified moments`);
    const gallery = document.createElement("div");
    gallery.className = "behavior-milestone-gallery";
    gallery.style.setProperty("--milestone-count", String(valid.length));
    const slots = new Map();
    for (const [index, frame] of valid.entries()) {
      const figure = document.createElement("figure");
      const replay = behaviorReplayFrameButton(frame, trajectoryUrl, trialId, "behavior-milestone-frame");
      const placeholder = document.createElement("span");
      placeholder.className = "behavior-milestone-placeholder";
      placeholder.textContent = "Rendering view";
      const order = document.createElement("span");
      order.className = "behavior-milestone-order";
      order.textContent = String(index + 1).padStart(2, "0");
      replay.append(placeholder, order);
      figure.append(replay, behaviorEvidenceCaption(frame));
      gallery.appendChild(figure);
      slots.set(Number(frame.index ?? index), { replay, order, frame });
    }
    section.append(header, gallery);
    container.appendChild(section);
    const captureToken = `${trajectoryUrl}|${valid.map((frame) => Number(frame.step || 0)).join(",")}`;
    section.dataset.captureToken = captureToken;
    void captureBehaviorMilestones({ visualScene, evidenceFrames: valid, trajectoryUrl, mechanisms })
      .then((captures) => {
        if (!section.isConnected || section.dataset.captureToken !== captureToken) return;
        for (const capture of captures) {
          const slot = slots.get(Number(capture.evidence_index));
          if (!slot || !capture.image_data_url) continue;
          const img = document.createElement("img");
          img.src = capture.image_data_url;
          img.alt = `Styled replay milestone ${Number(capture.evidence_index) + 1}`;
          behaviorReplaySelections.set(slot.replay, {
            attempt: Number(capture.trajectory_attempt || slot.frame.attempt || 1),
            step: Number(capture.trajectory_step || slot.frame.step || 0),
            view: capture.view || null,
            objects: capture.objects || null,
            mechanisms: capture.mechanisms || null,
            agent: slot.frame.agent || null,
          });
          slot.replay.replaceChildren(img, slot.order);
        }
        section.classList.add("ready");
      })
      .catch(() => {
        if (!section.isConnected || section.dataset.captureToken !== captureToken) return;
        section.classList.add("capture-error");
        for (const { replay } of slots.values()) {
          const placeholder = replay.querySelector(".behavior-milestone-placeholder");
          if (placeholder) placeholder.textContent = "Styled view unavailable";
        }
      });
  }

  if (!valid.length && !childSummary) return;
  const details = document.createElement("details");
  details.className = "behavior-run-details";
  details.open = !(valid.length && trajectoryUrl && visualScene);
  const summary = document.createElement("summary");
  summary.textContent = "Run details";
  const body = document.createElement("div");
  body.className = "behavior-run-detail-body";
  if (childSummary) {
    const notes = document.createElement("section");
    notes.className = "behavior-agent-notes";
    appendText(notes, "strong", "", "Agent notes");
    appendText(notes, "p", "behavior-child-summary", childSummary);
    body.appendChild(notes);
  }
  if (valid.length) {
    const policy = document.createElement("section");
    policy.className = "behavior-policy-section";
    const policyHead = document.createElement("div");
    policyHead.className = "behavior-run-detail-head";
    appendText(policyHead, "strong", "", "First-person policy view");
    appendText(policyHead, "span", "", `${valid.length} frames${replayVerified ? " · replay verified" : ""}`);
    const gallery = document.createElement("div");
    gallery.className = "behavior-evidence-gallery behavior-policy-gallery";
    for (const frame of valid) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = frame.url;
      img.alt = `First-person behavior evidence frame ${Number(frame.index || 0) + 1}`;
      img.loading = "lazy";
      let visual = img;
      if (trajectoryUrl) {
        const replay = behaviorReplayFrameButton(frame, trajectoryUrl, trialId, "behavior-evidence-frame");
        replay.appendChild(img);
        visual = replay;
      }
      figure.append(visual, behaviorEvidenceCaption(frame));
      gallery.appendChild(figure);
    }
    policy.append(policyHead, gallery);
    body.appendChild(policy);
  }
  details.append(summary, body);
  container.appendChild(details);
}

function behaviorReplayFrameButton(frame, trajectoryUrl, trialId, className) {
  const replay = document.createElement("button");
  replay.type = "button";
  replay.className = className;
  replay.dataset.behaviorAction = "replay-frame";
  replay.dataset.trialId = trialId;
  replay.dataset.trajectoryUrl = trajectoryUrl;
  replay.dataset.replayStep = String(Number(frame.step || 0));
  replay.dataset.replayAttempt = String(Number(frame.attempt || 1));
  behaviorReplaySelections.set(replay, {
    attempt: Number(frame.attempt || 1),
    step: Number(frame.step || 0),
    agent: frame.agent || null,
  });
  replay.title = "Show this milestone in the themed scene replay";
  replay.setAttribute("aria-label", `${frame.label || "Behavior milestone"}, step ${Number(frame.step || 0)}`);
  return replay;
}

function behaviorEvidenceCaption(frame) {
  const caption = document.createElement("figcaption");
  const fallbackEvent = frame.events?.at(-1)?.type
    ? String(frame.events.at(-1).type).replaceAll("_", " ")
    : "Observation";
  appendText(caption, "strong", "", frame.label || fallbackEvent);
  appendText(caption, "span", "", `Attempt ${Number(frame.attempt || 1)} · step ${Number(frame.step || 0)}`);
  return caption;
}

function renderStaleBehaviorHistory() {
  const details = document.createElement("details");
  details.className = "behavior-stale-history";
  const summary = document.createElement("summary");
  summary.textContent = "Previous run (outdated)";
  appendText(details, "p", "", "This result is retained for history only and is not counted as current evidence.");
  details.prepend(summary);
  els.behaviorTrialsBody.appendChild(details);
}

function appendBehaviorNotice(title, message, className) {
  const notice = document.createElement("div");
  notice.className = `behavior-trial-notice ${className || ""}`.trim();
  appendText(notice, "strong", "", title);
  appendText(notice, "span", "", message);
  els.behaviorTrialsBody.appendChild(notice);
}

function appendBehaviorStat(label, value, className) {
  const item = document.createElement("div");
  item.className = `check-stat ${className || ""}`.trim();
  appendText(item, "strong", "", String(value));
  appendText(item, "span", "", label);
  els.behaviorTrialStats.appendChild(item);
}

function renderProgressDots(container, label = "Running", visibleLabel = "") {
  container.replaceChildren();
  container.setAttribute("aria-label", label);
  if (visibleLabel) appendText(container, "span", "progress-label", visibleLabel);
  const dots = document.createElement("span");
  dots.className = "progress-dots";
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index += 1) dots.appendChild(document.createElement("span"));
  container.appendChild(dots);
}

function updateBehaviorTabBadge(summary, report, trialCount) {
  const status = String(summary?.status || "missing");
  const running = Boolean(
    hasServerActiveBehaviorRuns(summary)
    || behaviorRequestsForEnv(state.currentScene?.env_id).length,
  );
  const bad = ["failed", "needs_attention", "error"].includes(status);
  if (running) {
    renderProgressDots(els.behaviorTabBadge, "Agent tests running");
  } else {
    els.behaviorTabBadge.removeAttribute("aria-label");
    els.behaviorTabBadge.textContent = status === "not_applicable" ? "N/A" : String(trialCount || 0);
  }
  els.behaviorTabBadge.className = `tab-count ${running ? "live" : bad ? "bad" : ["stale", "partial"].includes(status) ? "warn" : report?.status === "passed" ? "good" : ""}`.trim();
}

function behaviorResultLabel(status, expected) {
  return {
    passed: "passed",
    inconclusive: "inconclusive",
    failed: expected === "should_not_succeed" ? "counterexample found" : "failed",
    invalid_setup: "invalid setup",
    error: "error",
    running: "running",
    ready: "ready",
  }[status] || status.replaceAll("_", " ");
}

function behaviorOutcomeSummary(result) {
  const status = String(result?.status || "");
  if (status === "passed" && result?.expected_outcome === "should_not_succeed") {
    return "The agent performed a valid bounded search without demonstrating the prohibited behavior.";
  }
  if (status === "passed") return "The agent completed the test and the simulator verified each required outcome.";
  if (status === "failed") return "The rollout demonstrated the prohibited counterexample, so the environment needs attention.";
  if (status === "invalid_setup") return "Preflight found that this setup cannot produce trustworthy behavioral evidence.";
  if (status === "inconclusive") return "The requested behavior was not demonstrated; this may reflect the policy, budget, or environment.";
  return "The behavior run ended without a conclusive typed result.";
}

function behaviorTerminationLabel(reason) {
  const normalized = String(reason || "finished");
  return {
    objective_satisfied: "Objective satisfied",
    step_budget: "Step budget reached",
    child_stopped: "Agent stopped",
    invalid_setup: "Invalid setup",
    counterexample_found: "Counterexample found",
  }[normalized] || normalized.replaceAll("_", " ");
}

async function handleBehaviorTrialAction(event) {
  const button = event.target.closest("[data-behavior-action]");
  if (!button) return;
  const trialId = button.dataset.trialId || "";
  if (button.dataset.behaviorAction === "run") {
    await runBehaviorTrials(trialId ? [trialId] : []);
  } else if (button.dataset.behaviorAction === "replay" && button.dataset.trajectoryUrl) {
    await startBehaviorReplay(button.dataset.trajectoryUrl, trialId);
  } else if (button.dataset.behaviorAction === "replay-frame" && button.dataset.trajectoryUrl) {
    const selection = behaviorReplaySelections.get(button) || {
      attempt: Number(button.dataset.replayAttempt || 1),
      step: Number(button.dataset.replayStep || 0),
    };
    await startBehaviorReplay(
      button.dataset.trajectoryUrl,
      trialId,
      selection,
    );
  }
}

async function runAllBehaviorTrials() {
  const trialIds = (state.currentScene?.env_behavior_trial_plan?.trials || [])
    .map((trial) => String(trial?.id || ""))
    .filter(Boolean);
  await Promise.allSettled(trialIds.map((trialId) => runBehaviorTrials([trialId])));
}

async function runBehaviorTrials(trialIds = []) {
  const scene = state.currentScene;
  if (!scene?.env_id) return;
  const plannedTrialIds = (scene.env_behavior_trial_plan?.trials || [])
    .map((trial) => String(trial?.id || ""))
    .filter(Boolean);
  const requestedTrialIds = uniqueBehaviorTrialIds(trialIds.length ? trialIds : plannedTrialIds);
  if (!requestedTrialIds.length) return;
  const currentRunView = buildBehaviorRunView({
    summary: scene.env_behavior_trials,
    trialIds: plannedTrialIds,
    localRunning: behaviorRequestsForEnv(scene.env_id).length > 0,
    requestedTrialIds: behaviorRequestsForEnv(scene.env_id).flatMap((request) => request.trialIds),
  });
  const activeTrialIds = new Set(currentRunView.activeTrialIds);
  if (requestedTrialIds.some((trialId) => activeTrialIds.has(trialId))) return;

  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  if (!behaviorRequestsForEnv(scene.env_id).length) state.behavior.activity = [];
  state.behavior.requests.set(requestId, {
    envId: scene.env_id,
    runId: "",
    trialIds: requestedTrialIds,
  });
  setInspectorTab("behavior");
  renderBehaviorTrials(scene);
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(scene.env_id)}/behavior-trials/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(withCodexModel({ trial_ids: requestedTrialIds, stream: true })),
    });
    if (!response.ok) throw new Error(`Behavior request failed: ${response.status}`);
    let donePayload = null;
    await consumeEventStream(response, {
      behavior_trials(payload) {
        handleBehaviorTrialProgress(payload, requestId, requestedTrialIds, scene.env_id);
      },
      done(payload) {
        donePayload = payload;
      },
    });
    if (!donePayload || donePayload.error || donePayload.isError) throw new Error(donePayload?.error || "Agent test failed");
    const latest = await fetchScene(scene.env_id).catch(() => donePayload.scene);
    if (latest) {
      mergeScene(latest);
      if (state.currentScene?.env_id === scene.env_id) state.currentScene = latest;
    }
  } catch (error) {
    appendBehaviorNotice("Agent test error", error.message || String(error), "bad");
  } finally {
    state.behavior.requests.delete(requestId);
    if (state.currentScene?.env_id === scene.env_id) {
      renderEnvHeader(state.currentScene);
      renderBehaviorTrials(state.currentScene);
    }
  }
}

function handleBehaviorTrialProgress(payload, requestId, requestedTrialIds, envId) {
  const request = state.behavior.requests.get(requestId);
  if (request && payload.run_id) request.runId = payload.run_id;
  state.behavior.activity.push({
    type: "behavior_trial",
    label: String(payload.status || "behavior").replaceAll("_", " "),
    message: payload.instruction || payload.error || "",
  });
  if (state.currentScene?.env_id !== envId) return;
  const previousSummary = state.currentScene.env_behavior_trials || {};
  const activeRuns = serverActiveBehaviorRuns(previousSummary).map((run) => ({ ...run }));
  const manifest = payload.manifest && typeof payload.manifest === "object" ? payload.manifest : {};
  const runId = String(payload.run_id || manifest.run_id || request?.runId || "");
  const previousRunIndex = activeRuns.findIndex((run) => String(run.run_id || "") === runId);
  const previousRun = previousRunIndex >= 0 ? activeRuns[previousRunIndex] : {};
  const trialIds = uniqueBehaviorTrialIds(
    manifest.trial_ids || previousRun.trial_ids || requestedTrialIds,
  );
  if (payload.trial_id && !trialIds.includes(payload.trial_id)) trialIds.push(payload.trial_id);
  const terminal = ["passed", "failed", "needs_attention", "partial", "error"].includes(
    String(payload.status || ""),
  );
  if (runId && terminal) {
    if (previousRunIndex >= 0) activeRuns.splice(previousRunIndex, 1);
  } else if (runId) {
    const updatedRun = {
      ...previousRun,
      ...manifest,
      run_id: runId,
      status: payload.status || previousRun.status || "running",
      trial_ids: trialIds,
    };
    if (previousRunIndex >= 0) activeRuns[previousRunIndex] = updatedRun;
    else activeRuns.push(updatedRun);
  }
  const activeRun = activeRuns.at(-1) || null;
  state.currentScene.env_behavior_trials = {
    ...previousSummary,
    status: activeRuns.length ? "running" : payload.status || previousSummary.status || "running",
    active_run: activeRun,
    active_runs: activeRuns,
    active_run_count: activeRuns.length,
    label: activeRuns.length
      ? "Behavior trials: running"
      : `Behavior trials: ${String(payload.status || "running").replaceAll("_", " ")}`,
  };
  if (payload.report) state.currentScene.env_behavior_trial_report = payload.report;
  renderEnvHeader(state.currentScene);
  renderBehaviorTrials(state.currentScene);
  setActivity("Agent Tests", state.behavior.activity);
}

async function dismissBehaviorTrials() {
  const scene = state.currentScene;
  if (!scene?.env_id || behaviorActiveForScene(scene)) return;
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(scene.env_id)}/behavior-trials/dismiss`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Request failed: ${response.status}`);
    const latest = data.scene || await fetchScene(scene.env_id);
    mergeScene(latest);
    state.currentScene = latest;
    renderBehaviorTrials(latest);
  } catch (error) {
    appendBehaviorNotice("Dismiss failed", error.message || String(error), "bad");
  }
}

async function repairFromBehaviorTrials() {
  const scene = state.currentScene;
  if (!scene?.env_id || state.running || behaviorActiveForScene(scene)) return;
  const regenerate = state.currentScene?.env_behavior_trials?.status === "stale"
    || Number(state.currentScene?.env_behavior_trial_report?.summary?.invalid_setup || 0) > 0;
  const displayMessage = regenerate
    ? "Regenerate behavior trials for the current environment without changing its geometry."
    : "Repair the environment using the current behavior trial evidence.";
  const priorHistory = Array.isArray(scene.history) ? scene.history.slice() : [];
  state.currentScene.history = [...priorHistory, { role: "user", content: displayMessage }];
  renderConversation(state.currentScene.history);
  const assistantNode = appendRevisionMessage("assistant", regenerate ? "Regenerating behavior trials..." : "Starting behavior-guided repair...");
  const activityEvents = [{ type: "request", message: displayMessage }];
  state.revising = true;
  setInspectorTab("activity");
  setRunning(true, "Repairing");
  try {
    const action = regenerate ? "regenerate" : "repair";
    const response = await fetch(`/api/scenes/${encodeURIComponent(scene.env_id)}/behavior-trials/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(withCodexModel({
        history: priorHistory,
        view_context: captureUserViewContext(),
        stream: true,
      })),
    });
    if (!response.ok) throw new Error(`Repair request failed: ${response.status}`);
    const data = await consumeRevisionStream(response, { activityEvents, assistantNode });
    if (!data || data.error || data.isError) throw new Error(data?.error || "Behavior-guided repair failed");
    const latest = await fetchScene(scene.env_id).catch(() => data.scene);
    if (latest) {
      mergeScene(latest);
      state.currentScene = latest;
      renderEnvPage();
      void ensureVisualReview(latest, data.visual_review_id);
    }
  } catch (error) {
    appendStreamLine(assistantNode, error.message || String(error));
  } finally {
    state.revising = false;
    setRunning(false);
  }
}

async function startTaskTrajectoryReplay(task, run) {
  try {
    await startBehaviorReplay(run.trajectory_url, `task run · ${task.task_id}`, null, {
      kind: "task",
      taskId: task.task_id,
      taskInstruction: task.instruction || "",
      runId: run.run_id,
      runNumber: Number(run.run_number || 0),
      activityUrl: run.activity_url || "",
      activity: Array.isArray(run.activity) ? run.activity : [],
    });
    if (state.behavior.replay.active && !state.behavior.replay.playing) toggleBehaviorReplay();
  } catch (error) {
    state.tasks.notice = { type: "error", message: error.message || String(error) };
    renderTasks(state.currentScene);
  }
}

async function loadTaskReplayActivity({ activityUrl = "", activity = [] } = {}) {
  if (!activityUrl) return normalizeTaskActivityTimeline(activity);
  try {
    const response = await fetch(activityUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`Agent activity request failed: ${response.status}`);
    const data = await response.json();
    return normalizeTaskActivityTimeline(data.events);
  } catch (error) {
    if (Array.isArray(activity) && activity.length) return normalizeTaskActivityTimeline(activity);
    throw error;
  }
}

async function startBehaviorReplay(url, trialId, target = null, context = {}) {
  if (!url || !state.currentScene) return;
  if (state.play.active) await stopPlayableEnvironment({ silent: true });
  const replay = state.behavior.replay;
  const loadToken = replay.loadToken + 1;
  const selection = Number.isFinite(target) ? { step: Number(target) } : target;
  if (!replay.active) replay.previousView = currentPreviewView();
  replay.loadToken = loadToken;
  const [response, taskActivity] = await Promise.all([
    fetch(url, { cache: "no-store" }),
    context.kind === "task" ? loadTaskReplayActivity(context) : Promise.resolve([]),
  ]);
  if (!response.ok) throw new Error(`Trajectory request failed: ${response.status}`);
  const data = await response.json();
  const frames = Array.isArray(data.frames) ? data.frames : [];
  if (!frames.length) throw new Error("Trajectory contains no replay frames");
  if (replay.loadToken !== loadToken) return;
  replay.active = true;
  replay.frames = frames;
  replay.index = 0;
  replay.trialId = trialId || data.trial_id || "trial";
  replay.kind = context.kind === "task" ? "task" : "behavior";
  replay.taskActivity = taskActivity;
  replay.taskId = String(context.taskId || "");
  replay.taskInstruction = String(context.taskInstruction || "");
  replay.runId = String(context.runId || "");
  replay.runNumber = Number(context.runNumber || 0);
  replay.overlayState = replay.kind === "task" ? {
    envId: state.currentScene.env_id,
    taskId: replay.taskId,
    runId: replay.runId,
    status: "replay",
    instruction: replay.taskInstruction,
    events: [],
    latestSceneFrame: null,
    isReplay: true,
    playing: false,
    runNumber: replay.runNumber,
  } : null;
  state.preview.mode = "visual";
  const previewReady = renderPreview();
  els.behaviorReplayControls.classList.remove("hidden");
  els.behaviorReplaySlider.min = "0";
  els.behaviorReplaySlider.max = String(frames.length - 1);
  const initialIndex = Number.isFinite(Number(selection?.step))
    ? nearestTrajectoryFrameIndex(frames, Number(selection.step), selection.attempt)
    : 0;
  await previewReady;
  if (replay.loadToken !== loadToken || !replay.active) return;
  if (replay.overlayState) mountTaskRunOverlay(replay.overlayState);
  setBehaviorReplayFrame(Math.max(0, initialIndex), selection);
}

function setBehaviorReplayFrame(index, selection = null) {
  const replay = state.behavior.replay;
  if (!replay.active || !replay.frames.length) return;
  replay.index = clamp(Math.round(index), 0, replay.frames.length - 1);
  const frame = replay.frames[replay.index];
  const frameState = behaviorReplayFrameState(frame, selection);
  if (frameState.view) {
    state.preview.visual = clonePreviewView({ ...state.preview.visual, ...frameState.view });
    state.preview.renderer?.setView(state.preview.visual);
  }
  state.preview.renderer?.applyPhysicsState(frameState.objects);
  state.preview.renderer?.applyGameState?.({ mechanisms: frameState.mechanisms });
  els.behaviorReplaySlider.value = String(replay.index);
  els.behaviorReplayLabel.textContent = `${replay.index + 1} / ${replay.frames.length} · ${replay.trialId}`;
  if (replay.kind === "task") updateTaskTrajectoryReplayOverlay(replay, frame);
}

function toggleBehaviorReplay() {
  const replay = state.behavior.replay;
  if (!replay.active) return;
  replay.playing = !replay.playing;
  els.behaviorReplayToggle.textContent = replay.playing ? "Pause" : "Play";
  if (replay.overlayState) {
    replay.overlayState.playing = replay.playing;
    updateTaskRunOverlay(replay.overlayState);
  }
  if (replay.timer !== null) window.clearInterval(replay.timer);
  replay.timer = replay.playing ? window.setInterval(() => {
    const next = replay.index + 1;
    if (next >= replay.frames.length) {
      replay.playing = false;
      window.clearInterval(replay.timer);
      replay.timer = null;
      els.behaviorReplayToggle.textContent = "Play";
      if (replay.overlayState) {
        replay.overlayState.playing = false;
        updateTaskRunOverlay(replay.overlayState);
      }
      return;
    }
    setBehaviorReplayFrame(next);
  }, 50) : null;
}

function exitBehaviorReplay() {
  const replay = state.behavior.replay;
  if (replay.timer !== null) window.clearInterval(replay.timer);
  const previousView = replay.previousView;
  Object.assign(replay, {
    active: false,
    frames: [],
    index: 0,
    playing: false,
    timer: null,
    trialId: "",
    loadToken: replay.loadToken + 1,
    previousView: null,
    kind: "behavior",
    taskActivity: [],
    taskId: "",
    taskInstruction: "",
    runId: "",
    runNumber: 0,
    overlayState: null,
  });
  els.taskRunOverlay.classList.add("hidden");
  els.previewStage.classList.remove("is-task-running", "is-task-replay");
  delete els.previewStage.dataset.taskStep;
  els.behaviorReplayControls.classList.add("hidden");
  els.behaviorReplayToggle.textContent = "Play";
  if (previousView) state.preview.visual = previousView;
  if (state.route.page === "env") renderPreview();
}

function currentPreviewView() {
  return clonePreviewView({
    ...state.preview.visual,
    ...(state.preview.renderer?.view || {}),
  });
}

function clonePreviewView(view) {
  const copy = { ...(view || {}) };
  if (Array.isArray(copy.target)) copy.target = copy.target.map(Number);
  return copy;
}

function appendCheckStat(label, value, className) {
  const item = document.createElement("div");
  item.className = `check-stat ${className || ""}`.trim();
  const count = document.createElement("strong");
  count.textContent = String(value);
  const name = document.createElement("span");
  name.textContent = label;
  item.append(count, name);
  els.envCheckStats.appendChild(item);
}

function compactMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return "";
  const text = JSON.stringify(metrics, null, 2);
  return text.length > 700 ? `${text.slice(0, 700)}\n...` : text;
}

function renderSceneActivity(scene) {
  const activity = latestToolActivity(scene?.history || []);
  const reviewActivity = state.visualReview.activityByEnv.get(scene?.env_id) || [];
  const combined = [...activity, ...reviewActivity];
  setActivity(reviewActivity.length ? "Visual Review" : activity.length ? "Last Run" : "Ready", combined);
}

function latestToolActivity(history) {
  if (!Array.isArray(history)) return [];
  for (const turn of history.slice().reverse()) {
    if (!Array.isArray(turn?.activity)) continue;
    const activity = turn.activity.filter((event) => {
      const type = String(event?.type || "").toLowerCase();
      return type !== "agent_message";
    });
    if (activity.length) return activity;
  }
  return [];
}

function renderConversation(history) {
  els.revisionMessages.replaceChildren();
  const turns = Array.isArray(history) ? history : [];
  if (!turns.length) {
    appendRevisionMessage("assistant", state.currentScene ? "Ask for changes to this environment." : "Select an environment, then ask for changes here.");
    return;
  }
  for (const turn of turns) {
    if (!turn?.role || !turn?.content) continue;
    appendRevisionMessage(turn.role === "user" ? "user" : "assistant", conversationContentForTurn(turn));
  }
  els.revisionMessages.scrollTop = els.revisionMessages.scrollHeight;
}

function appendRevisionMessage(role, content) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const roleNode = document.createElement("div");
  roleNode.className = "role";
  roleNode.textContent = role === "user" ? "You" : "Codex";
  const contentNode = document.createElement("div");
  contentNode.className = "content";
  setConversationContent(contentNode, role, content);
  node.append(roleNode, contentNode);
  els.revisionMessages.appendChild(node);
  els.revisionMessages.scrollTop = els.revisionMessages.scrollHeight;
  return contentNode;
}

function setConversationContent(node, role, content) {
  const rawContent = productDisplayText(content);
  node.dataset.rawContent = rawContent;
  const blocks = conversationBlocks(role, rawContent);
  if (blocks.length <= 1) {
    node.textContent = blocks[0] || "";
    return;
  }
  node.replaceChildren();
  const visibleBlockCount = 4;
  const collapseEarlier = role === "assistant" && blocks.length > visibleBlockCount + 1;
  if (collapseEarlier) {
    const earlierBlocks = blocks.slice(0, -visibleBlockCount);
    const details = document.createElement("details");
    details.className = "conversation-earlier";
    const summary = document.createElement("summary");
    summary.textContent = `Earlier updates (${earlierBlocks.length})`;
    const body = document.createElement("div");
    body.className = "conversation-earlier-body";
    appendConversationBlocks(body, earlierBlocks);
    details.append(summary, body);
    node.appendChild(details);
  }
  appendConversationBlocks(node, collapseEarlier ? blocks.slice(-visibleBlockCount) : blocks);
}

function productDisplayText(content) {
  return String(content || "").trim();
}

function appendConversationBlocks(node, blocks) {
  for (const block of blocks) appendText(node, "p", "conversation-block", block);
}

function resizeRevisionPrompt() {
  const prompt = els.revisionPrompt;
  if (!prompt) return;
  prompt.style.height = "auto";
  const targetHeight = Math.min(140, Math.max(54, prompt.scrollHeight));
  prompt.style.height = `${targetHeight}px`;
  prompt.style.overflowY = prompt.scrollHeight > targetHeight ? "auto" : "hidden";
}

function conversationBlocks(role, content) {
  const text = String(content || "").trim();
  if (!text) return [];
  const explicitBlocks = text.split(/\n+/).map((part) => part.trim()).filter(Boolean);
  if (explicitBlocks.length > 1 || role !== "assistant" || text.length < 420) return explicitBlocks;
  const sentences = text.match(/[^.!?]+(?:[.!?]+(?=\s|$)|$)/g) || [text];
  const blocks = [];
  let current = "";
  for (const sentence of sentences) {
    const value = sentence.trim();
    if (!value) continue;
    if (current && current.length + value.length > 320) {
      blocks.push(current);
      current = value;
    } else {
      current = current ? `${current} ${value}` : value;
    }
  }
  if (current) blocks.push(current);
  return blocks;
}

function conversationContentForTurn(turn) {
  const content = String(turn?.content || "");
  if (!isGenericAssistantFallback(content)) return content;
  const publicUpdates = (turn.activity || [])
    .filter((event) => String(event?.type || "").toLowerCase() === "agent_message")
    .map((event) => String(event.message || "").trim())
    .filter(Boolean);
  return publicUpdates.length ? publicUpdates.join("\n") : content;
}

function isGenericAssistantFallback(content) {
  return [
    "Created and finalized the environment.",
    "Applied the requested changes and finalized the environment.",
  ].includes(String(content || "").trim());
}

function handlePreviewWheel(event) {
  if (state.route.page !== "env") return;
  event.preventDefault();
  if (state.preview.mode === "visual") {
    setVisualDistance(state.preview.visual.distance * (event.deltaY > 0 ? 1.1 : 0.9));
    return;
  }
  if (!state.preview.url) return;
  const direction = event.deltaY > 0 ? -1 : 1;
  setPhysicsZoom(state.preview.zoom + direction * PREVIEW_ZOOM_STEP, event);
}

function startPreviewDrag(event) {
  if (!state.currentScene || state.route.page !== "env") return;
  if (state.preview.mode === "visual") {
    state.preview.dragMode = state.play.active
      ? "rotate"
      : visualIsZoomed() && event.shiftKey
        ? "pan"
        : "rotate";
  } else {
    if (!state.preview.url) return;
    const canRotate = state.preview.orbitFrames.length > 1;
    if (!canRotate && state.preview.zoom <= 1) return;
    state.preview.dragMode = canRotate ? "rotate" : "pan";
  }
  state.preview.dragging = true;
  state.preview.pointerId = event.pointerId;
  state.preview.lastX = event.clientX;
  state.preview.lastY = event.clientY;
  els.previewStage.classList.add("is-panning");
  els.previewStage.setPointerCapture(event.pointerId);
}

function movePreviewDrag(event) {
  if (!state.preview.dragging || state.preview.pointerId !== event.pointerId) return;
  const dx = event.clientX - state.preview.lastX;
  const dy = event.clientY - state.preview.lastY;
  state.preview.lastX = event.clientX;
  state.preview.lastY = event.clientY;
  if (state.preview.mode === "visual") {
    if (state.preview.dragMode === "pan") {
      const scale = state.preview.visual.distance / 450;
      state.preview.visual.panX -= dx * scale;
      state.preview.visual.panY += dy * scale;
    } else {
      state.preview.visual.azimuth += dx * 0.35;
      state.preview.visual.elevation += dy * 0.12;
    }
    applyVisualView();
    return;
  }
  if (state.preview.dragMode === "rotate") {
    const steps = Math.trunc(dx / 24);
    if (steps !== 0) {
      setOrbitIndex(state.preview.orbitIndex + steps);
      state.preview.lastX -= dx - steps * 24;
    }
    return;
  }
  state.preview.x += dx;
  state.preview.y += dy;
  updatePhysicsTransform();
}

function stopPreviewDrag(event) {
  if (event.pointerId !== undefined && state.preview.pointerId !== event.pointerId) return;
  state.preview.dragging = false;
  state.preview.dragMode = "";
  state.preview.pointerId = null;
  els.previewStage.classList.remove("is-panning");
}

function resetPreviewState(sceneId, sceneVersion = "") {
  state.preview.sceneId = sceneId;
  state.preview.sceneVersion = sceneVersion;
  state.preview.url = "";
  state.preview.zoom = 1;
  state.preview.x = 0;
  state.preview.y = 0;
  state.preview.orbitFrames = [];
  state.preview.orbitIndex = 0;
  state.preview.dragging = false;
  state.preview.dragMode = "";
  state.preview.pointerId = null;
  resetVisualViewFromScene(state.currentScene?.visual_scene || null);
}

function previewVersionKey(scene) {
  if (!scene) return "";
  const objectSignature = (scene.objects || []).map((object) => [
    object.id,
    object.semantic_type,
    object.shape,
    object.body_type,
    object.position,
    object.size,
    object.yaw,
  ]);
  return JSON.stringify({
    env_id: scene.env_id || "",
    visual_scene_url: scene.visual_scene_url || "",
    previews: scene.previews || {},
    orbit_previews: scene.orbit_previews || [],
    objects: objectSignature,
  });
}

function resetVisualViewFromScene(visualScene) {
  const camera = visualScene?.camera || {};
  const distance = Number(camera.distance || 14);
  state.preview.visual = {
    azimuth: Number(camera.azimuth ?? -42),
    elevation: Number(camera.elevation ?? 38),
    distance,
    defaultDistance: distance,
    panX: 0,
    panY: 0,
  };
}

function applyVisualView() {
  const camera = state.preview.visualScene?.camera || {};
  state.preview.visual.distance = clamp(
    state.preview.visual.distance,
    Number(camera.min_distance || 4),
    Number(camera.max_distance || 40),
  );
  state.preview.visual.elevation = clamp(state.preview.visual.elevation, 18, 72);
  if (state.play.active) {
    state.preview.visual.panX = 0;
    state.preview.visual.panY = 0;
    els.previewStage.dataset.cameraAzimuth = String(roundNumber(wrapDegrees(state.preview.visual.azimuth)));
    els.previewStage.dataset.cameraElevation = String(roundNumber(state.preview.visual.elevation));
  }
  state.preview.renderer?.setView(state.preview.visual);
}

function setVisualDistance(distance) {
  state.preview.visual.distance = distance;
  applyVisualView();
}

function setPhysicsZoom(nextZoom, anchor = null) {
  if (!state.preview.url) return;
  const previousZoom = state.preview.zoom;
  const zoom = clamp(nextZoom, MIN_PREVIEW_ZOOM, MAX_PREVIEW_ZOOM);
  if (zoom === previousZoom) return;
  if (anchor) {
    const rect = els.previewStage.getBoundingClientRect();
    const dx = anchor.clientX - rect.left - rect.width / 2 - state.preview.x;
    const dy = anchor.clientY - rect.top - rect.height / 2 - state.preview.y;
    const ratio = zoom / previousZoom;
    state.preview.x -= dx * (ratio - 1);
    state.preview.y -= dy * (ratio - 1);
  }
  state.preview.zoom = zoom;
  updatePhysicsTransform();
}

function setOrbitIndex(nextIndex) {
  const frameCount = state.preview.orbitFrames.length;
  if (frameCount <= 1) return;
  state.preview.orbitIndex = ((Math.round(nextIndex) % frameCount) + frameCount) % frameCount;
  renderPhysicsPreview(state.currentScene);
}

function updatePhysicsTransform() {
  clampPhysicsPan();
  const image = els.previewStage.querySelector(".preview-image");
  if (image) {
    image.style.transform = `translate(${state.preview.x}px, ${state.preview.y}px) scale(${state.preview.zoom})`;
  }
}

function clampPhysicsPan() {
  if (state.preview.zoom <= 1) {
    state.preview.x = 0;
    state.preview.y = 0;
    return;
  }
  const rect = els.previewStage.getBoundingClientRect();
  const maxX = rect.width * (state.preview.zoom - 1) / 2;
  const maxY = rect.height * (state.preview.zoom - 1) / 2;
  state.preview.x = clamp(state.preview.x, -maxX, maxX);
  state.preview.y = clamp(state.preview.y, -maxY, maxY);
}

function physicsPreviewUrlFor(scene) {
  if (state.preview.orbitFrames.length) {
    return state.preview.orbitFrames[state.preview.orbitIndex] || state.preview.orbitFrames[0] || "";
  }
  if (!scene?.previews) return "";
  return scene.previews.overview || Object.values(scene.previews)[0] || "";
}

function orbitFramesFor(scene) {
  if (!Array.isArray(scene?.orbit_previews)) return [];
  return scene.orbit_previews.filter((url) => typeof url === "string" && url);
}

function disposeVisualRenderer() {
  if (state.preview.renderer) {
    state.preview.renderer.dispose();
    state.preview.renderer = null;
  }
  state.preview.ready = Promise.resolve(false);
}

function ensureCreatePreview() {
  if (state.createPreview.renderer || !els.createPreviewStage) return;
  els.createPreviewStage.replaceChildren();
  const renderer = new VisualPreviewRenderer(els.createPreviewStage);
  state.createPreview.renderer = renderer;
  state.createPreview.ready = (async () => {
    const response = await fetch("/api/courtyard-baseline", { cache: "no-store" });
    if (!response.ok) throw new Error(`Courtyard baseline request failed: ${response.status}`);
    const baseline = await response.json();
    if (state.createPreview.renderer !== renderer) return;
    state.createPreview.visualScene = baseline.visual_scene;
    state.createPreview.spec = baseline.spec;
    const camera = baseline.visual_scene?.camera || {};
    const distance = Number(camera.distance || 17);
    state.createPreview.view = {
      target: camera.target || [0, 0, 0.5],
      distance,
      defaultDistance: distance,
      azimuth: Number(camera.azimuth ?? -42),
      elevation: Number(camera.elevation ?? 38),
      panX: 0,
      panY: 0,
    };
    await renderer.setScene(baseline.visual_scene);
    applyCreatePreviewView();
  })().catch(() => {
    if (state.createPreview.renderer !== renderer) return;
    renderer.dispose();
    state.createPreview.renderer = null;
    state.createPreview.ready = null;
    state.createPreview.visualScene = null;
    state.createPreview.spec = null;
    state.createPreview.view = null;
    els.createPreviewStage.textContent = "Courtyard preview unavailable.";
  });
}

function disposeCreatePreviewRenderer() {
  state.createPreview.renderer?.dispose();
  state.createPreview.renderer = null;
  state.createPreview.ready = null;
  state.createPreview.visualScene = null;
  state.createPreview.spec = null;
  state.createPreview.view = null;
  stopCreatePreviewDrag({});
}

function applyCreatePreviewView() {
  const view = state.createPreview.view;
  if (!view) return;
  const camera = state.createPreview.visualScene?.camera || {};
  view.distance = clamp(view.distance, Number(camera.min_distance || 4), Number(camera.max_distance || 40));
  view.elevation = clamp(view.elevation, 18, 72);
  state.createPreview.renderer?.setView(view);
  els.createPreviewStage.dataset.cameraAzimuth = String(roundNumber(wrapDegrees(view.azimuth)));
  els.createPreviewStage.dataset.cameraElevation = String(roundNumber(view.elevation));
  els.createPreviewStage.dataset.cameraDistance = String(roundNumber(view.distance));
  els.createPreviewStage.dataset.cameraPanX = String(roundNumber(view.panX));
  els.createPreviewStage.dataset.cameraPanY = String(roundNumber(view.panY));
}

function changeCreatePreviewZoom(factor) {
  if (!state.createPreview.view || state.running) return;
  state.createPreview.view.distance *= factor;
  applyCreatePreviewView();
}

function handleCreatePreviewWheel(event) {
  if (state.route.page !== "create" || !state.createPreview.view || state.running) return;
  event.preventDefault();
  changeCreatePreviewZoom(event.deltaY > 0 ? 1.1 : 0.9);
}

function createPreviewIsZoomed() {
  const view = state.createPreview.view;
  return Boolean(view && view.distance < view.defaultDistance * 0.96);
}

function startCreatePreviewDrag(event) {
  if (state.route.page !== "create" || !state.createPreview.view || state.running) return;
  state.createPreview.dragMode = createPreviewIsZoomed() && event.shiftKey ? "pan" : "rotate";
  state.createPreview.dragging = true;
  state.createPreview.pointerId = event.pointerId;
  state.createPreview.lastX = event.clientX;
  state.createPreview.lastY = event.clientY;
  els.createPreviewStage.classList.add("is-panning");
  els.createPreviewStage.setPointerCapture(event.pointerId);
}

function moveCreatePreviewDrag(event) {
  if (!state.createPreview.dragging || state.createPreview.pointerId !== event.pointerId) return;
  const dx = event.clientX - state.createPreview.lastX;
  const dy = event.clientY - state.createPreview.lastY;
  state.createPreview.lastX = event.clientX;
  state.createPreview.lastY = event.clientY;
  if (state.createPreview.dragMode === "pan") {
    const scale = state.createPreview.view.distance / 450;
    state.createPreview.view.panX -= dx * scale;
    state.createPreview.view.panY += dy * scale;
  } else {
    state.createPreview.view.azimuth += dx * 0.35;
    state.createPreview.view.elevation += dy * 0.12;
  }
  applyCreatePreviewView();
}

function stopCreatePreviewDrag(event) {
  if (event.pointerId !== undefined && state.createPreview.pointerId !== event.pointerId) return;
  state.createPreview.dragging = false;
  state.createPreview.dragMode = "";
  state.createPreview.pointerId = null;
  els.createPreviewStage.classList.remove("is-panning");
}

function visualIsZoomed() {
  return state.preview.visual.distance < state.preview.visual.defaultDistance * 0.96;
}

function setRunning(value, label = "Generating") {
  state.running = value;
  els.generateButton.disabled = value;
  els.newEnvButton.disabled = value;
  els.homeCreateButton.disabled = value;
  els.emptyCreateButton.disabled = value;
  els.revisionPrompt.disabled = value || state.play.active || !state.currentScene;
  els.revisionButton.disabled = value || state.play.active || !state.currentScene;
  els.generateButton.textContent = value ? label : "Generate";
  updatePlayButton(state.currentScene);
  renderAgentPanelHeader();
}

function setStatus(label, className) {
  els.topStatus.textContent = label;
  els.topStatus.className = `status ${className}`;
}

function setActivity(label, events) {
  renderActivityLog(els.envActivityState, els.envActivityLog, label, events);
  const visibleCount = (Array.isArray(events) ? events : []).filter((event) => {
    return String(event?.type || "").toLowerCase() !== "agent_message";
  }).length;
  els.activityTabBadge.textContent = String(visibleCount);
  els.activityTabBadge.className = `tab-count ${state.running ? "live" : ""}`.trim();
}

function renderActivityLog(stateNode, logNode, label, events) {
  if (!stateNode || !logNode) return;
  stateNode.textContent = label;
  logNode.replaceChildren();
  const visibleEvents = (Array.isArray(events) ? events : []).filter((event) => {
    const type = String(event?.type || "").toLowerCase();
    return type !== "agent_message";
  });
  if (!visibleEvents.length) {
    appendText(logNode, "div", "empty-list", "No activity.");
    return;
  }
  for (const event of visibleEvents.slice(-80)) {
    const item = document.createElement("div");
    item.className = activityItemClass(event);
    const title = document.createElement("strong");
    title.textContent = formatActivityLabel(event);
    const body = document.createElement("span");
    body.textContent = activityMessage(event);
    item.append(title, body);
    logNode.appendChild(item);
  }
}

function activityItemClass(event) {
  const type = String(event?.type || "").toLowerCase();
  const isError = Boolean(event?.isError) || type.includes("error") || String(event?.message || "").toLowerCase().includes("error:");
  return `activity-item ${isError ? "error" : ""}`.trim();
}

function activityMessage(event) {
  const message = String(event?.message || "");
  if (message) return message;
  if (event?.input) return JSON.stringify(event.input, null, 2);
  if (event?.result) return String(event.result).slice(0, 700);
  return "";
}

function formatActivityLabel(event) {
  const type = String(event?.type || "").toLowerCase();
  const label = String(event?.label || "");
  if (type === "agent_message" || label.toLowerCase() === "agent message") return "Codex update";
  if (type.includes("tool_call") || label.toLowerCase().includes("tool call")) return event?.name ? `Tool: ${event.name}` : "Tool call";
  if (type.includes("tool_result") || label.toLowerCase().includes("tool result")) return event?.name ? `Tool result: ${event.name}` : "Tool result";
  return label || event?.name || event?.type || "Progress";
}

function sceneStatusLabel(scene) {
  if (scene?.status === "generating") return "Building";
  if (scene?.status === "finalized") return "Finalized";
  return "Draft";
}

function sceneStatusClass(scene) {
  if (scene?.status === "generating") return "running";
  if (scene?.status === "finalized") return "done";
  return "idle";
}

function appendText(parent, tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

function wrapDegrees(value) {
  return ((Math.round(value) + 180) % 360 + 360) % 360 - 180;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function roundNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  const scale = 10 ** digits;
  return Math.round(number * scale) / scale;
}

function roundVector(value) {
  if (!Array.isArray(value)) return null;
  const rounded = value.map((item) => roundNumber(item));
  return rounded.every((item) => item !== null) ? rounded : null;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
