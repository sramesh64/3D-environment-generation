const DEFAULT_TIMESTEP_SECONDS = 0.01;
const MAX_REALTIME_ADVANCE_SECONDS = 0.1;

export function recordingFrameCount({ elapsedMs, timestep = DEFAULT_TIMESTEP_SECONDS } = {}) {
  const safeTimestep = positiveNumber(timestep, DEFAULT_TIMESTEP_SECONDS);
  const elapsedSeconds = Math.max(
    safeTimestep,
    Math.min(MAX_REALTIME_ADVANCE_SECONDS, finiteNumber(elapsedMs, 0) / 1000),
  );
  return Math.max(1, Math.round(elapsedSeconds / safeTimestep));
}

export function recordingReportRequested({ activeInput, idleTicks, idleLimit } = {}) {
  const ticks = Math.max(0, Math.floor(finiteNumber(idleTicks, 0)));
  const limit = Math.max(1, Math.floor(positiveNumber(idleLimit, 1)));
  return !activeInput && ticks === limit;
}

export function recordingFinishView({ active, hasActions, finishPending, finishing } = {}) {
  const busy = Boolean(finishPending || finishing);
  return {
    disabled: !active || !hasActions || busy,
    label: busy ? "Finishing..." : "Finish Recording",
  };
}

export function recordingFinishIntent({ active, sessionId, hasActions, finishing, requestInFlight } = {}) {
  const accepted = Boolean(active && sessionId && hasActions && !finishing);
  return {
    accepted,
    startImmediately: accepted && !requestInFlight,
  };
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function positiveNumber(value, fallback) {
  const number = finiteNumber(value, fallback);
  return number > 0 ? number : fallback;
}
