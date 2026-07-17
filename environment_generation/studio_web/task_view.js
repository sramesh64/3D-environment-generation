export function buildTaskTestRows(task, report = null) {
  const reportTests = new Map((report?.tests || []).map((item) => [String(item.id || ""), item]));
  const rows = [];
  for (const test of task?.tests || []) {
    const testId = String(test.id || "");
    const conditions = Array.isArray(test.conditions) ? test.conditions : [];
    const results = new Map(
      (reportTests.get(testId)?.conditions || []).map((item) => [String(item.id || ""), item]),
    );
    if (!conditions.length) {
      rows.push({
        id: testId,
        testId,
        description: readableText(test.description, testId || "Trajectory test"),
        passed: null,
      });
      continue;
    }
    for (const condition of conditions) {
      const conditionId = String(condition.id || "");
      const result = results.get(conditionId);
      rows.push({
        id: conditionId || testId,
        testId,
        description: readableText(
          condition.description,
          conditions.length === 1 ? test.description : conditionId || condition.predicate?.type || "Condition",
        ),
        passed: result ? Boolean(result.passed) : null,
      });
    }
  }
  return rows;
}

export function buildTaskResultView(task, selection = null) {
  const report = selection?.report && !selection.loading && !selection.error
    ? selection.report
    : null;
  const rows = buildTaskTestRows(task, report);
  const passed = rows.filter((row) => row.passed === true).length;
  const label = String(selection?.label || "").trim();
  let summary = `${rows.length} ${rows.length === 1 ? "test" : "tests"}`;
  if (selection?.loading) summary = `${label || "Result"} · loading`;
  else if (selection?.error) summary = `${label || "Result"} · unavailable`;
  else if (report) summary = `${label ? `${label} · ` : ""}${passed}/${rows.length} passed`;
  return { rows, report, passed, summary };
}

function readableText(value, fallback) {
  const text = String(value || "").trim();
  return text || String(fallback || "").trim() || "Condition";
}
