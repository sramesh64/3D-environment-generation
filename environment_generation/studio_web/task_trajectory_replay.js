function finiteStep(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.round(number)) : null;
}

function runNumber(run, fallback) {
  const match = /^run-(\d+)/.exec(String(run?.run_id || ""));
  return match ? Number(match[1]) : fallback;
}

export function normalizeTaskActivityTimeline(events) {
  let currentStep = 0;
  return (Array.isArray(events) ? events : [])
    .filter((event) => event && typeof event === "object")
    .map((event, index) => {
      const explicit = finiteStep(
        event.step ?? event.summary?.step ?? event.step_count,
      );
      if (explicit !== null) currentStep = Math.max(currentStep, explicit);
      return {
        ...event,
        sequence: Number.isFinite(Number(event.sequence)) ? Number(event.sequence) : index + 1,
        step: currentStep,
      };
    });
}

export function taskActivityAtStep(timeline, step) {
  const target = finiteStep(step) ?? 0;
  return (Array.isArray(timeline) ? timeline : []).filter(
    (event) => finiteStep(event?.step) <= target,
  );
}

export function taskTrajectoryRuns(task) {
  const summaries = Array.isArray(task?.run_summaries) ? task.run_summaries : [];
  const values = summaries.map((run, index) => ({
    ...run,
    run_number: runNumber(run, index + 1),
  }));
  const latest = task?.latest_run;
  if (latest?.run_id) {
    const index = values.findIndex((run) => run.run_id === latest.run_id);
    const merged = {
      ...(index >= 0 ? values[index] : {}),
      ...latest,
      run_number: runNumber(latest, index >= 0 ? index + 1 : values.length + 1),
    };
    if (index >= 0) values[index] = merged;
    else values.push(merged);
  }
  return values
    .filter((run) => run.run_id && run.trajectory_url)
    .sort((left, right) => {
      const leftTime = Date.parse(left.completed_at || left.created_at || "") || 0;
      const rightTime = Date.parse(right.completed_at || right.created_at || "") || 0;
      return rightTime - leftTime || Number(right.run_number) - Number(left.run_number);
    });
}
