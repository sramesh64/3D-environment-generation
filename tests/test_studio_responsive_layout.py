from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Error, Page, Playwright, sync_playwright


WEB_ROOT = Path(__file__).resolve().parents[1] / "environment_generation" / "studio_web"

LAYOUT_FIXTURE = """
<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"></head>
  <body>
    <main class="app">
      <header class="topbar">
        <button class="brand" type="button">
          <span class="brand-mark">3D</span>
          <span><strong>Environment Generation</strong><span>1 environment</span></span>
        </button>
        <div class="top-actions">
          <label class="model-picker">
            <span class="model-picker-label">Model</span>
            <select><option>gpt-5</option></select>
          </label>
          <div class="status">Ready</div>
          <button class="ghost" type="button">Primitives</button>
          <button type="button">New Environment</button>
        </div>
      </header>
      <section class="page env-page">
        <div class="env-head">
          <div class="env-identity"><h1>Responsive test</h1></div>
        </div>
        <div class="env-layout">
          <div class="workspace-main">
            <section class="panel stage-panel">
              <div class="preview-stage"></div>
            </section>
            <section class="panel inspector-panel">
              <div class="inspector-tabs">
                <button class="inspector-tab active" type="button">Objects</button>
              </div>
              <div class="object-list">
                <div class="object-row object-row-head">
                  <span>Object</span><span>Type</span><span>Position</span><span>Size</span>
                </div>
                <div class="object-row">
                  <strong>crate</strong>
                  <span data-label="Type">pushable box</span>
                  <span data-label="Position">1, 2, 0.5</span>
                  <span data-label="Size">1 x 1 x 1</span>
                </div>
              </div>
            </section>
          </div>
          <aside class="inspect-column">
            <section class="panel conversation-panel">
              <div class="agent-header"><h3>Edit With Agent</h3></div>
            </section>
          </aside>
        </div>
      </section>
    </main>
  </body>
</html>
"""

TASK_LAYOUT_FIXTURE = """
<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"></head>
  <body>
    <main class="app">
      <section class="panel inspector-panel task-fixture">
        <div class="task-list">
          <article class="task-card">
            <div class="task-card-head">
              <div class="task-card-title"><strong>Reach the charging goal without touching a hazard.</strong></div>
              <div class="task-card-head-side">
                <span class="task-status validated">Validated</span>
                <div class="task-card-actions">
                  <div class="task-agent-controls">
                    <select class="task-agent-model"><option>Default model</option></select>
                    <button class="primary" type="button">Run Agent</button>
                  </div>
                  <details class="task-action-menu">
                    <summary class="task-action-menu-trigger" aria-label="More task actions">...</summary>
                    <div class="task-secondary-actions task-action-menu-items">
                      <button class="ghost" type="button">Rerecord Oracle</button>
                      <button class="ghost" type="button">Replay Oracle</button>
                      <button class="ghost danger" type="button">Delete</button>
                    </div>
                  </details>
                </div>
              </div>
            </div>
            <div class="task-card-content">
              <section class="task-tests">
                <div class="task-tests-head"><strong>Tests</strong><span>3/3 passed</span></div>
                <div class="task-test-list">
                  <div class="task-condition pass"><span class="task-condition-dot"></span><span>Robot enters the goal.</span></div>
                  <div class="task-condition pass"><span class="task-condition-dot"></span><span>Robot avoids hazards.</span></div>
                  <div class="task-condition pass"><span class="task-condition-dot"></span><span>Robot stays in bounds.</span></div>
                </div>
              </section>
              <details class="task-trajectory-history">
                <summary class="task-trajectory-head"><strong>Agent runs</strong><span>8 saved</span></summary>
                <div class="task-trajectory-list">
                  <div class="task-trajectory-row fail">
                    <div class="task-trajectory-identity">
                      <div><span class="task-trajectory-dot"></span><strong>Run 8</strong><span class="task-trajectory-status">Did not pass</span></div>
                      <div class="task-trajectory-meta"><span>1752 steps</span><span>73 actions</span><span>gpt-5.5</span></div>
                    </div>
                    <button class="ghost task-replay-button" type="button">Replay</button>
                  </div>
                </div>
              </details>
            </div>
          </article>
        </div>
      </section>
    </main>
  </body>
</html>
"""


@pytest.fixture(scope="module")
def layout_browser() -> Iterator[Browser]:
    playwright: Playwright = sync_playwright().start()
    browser: Browser | None = None
    try:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Playwright Chromium is not installed")
            raise
        yield browser
    finally:
        if browser is not None:
            browser.close()
        playwright.stop()


@pytest.fixture(scope="module")
def responsive_page(layout_browser: Browser) -> Iterator[Page]:
    stylesheet = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
    page = layout_browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        page.set_content(f"<style>{stylesheet}</style>{LAYOUT_FIXTURE}")
        yield page
    finally:
        page.close()


@pytest.fixture(scope="module")
def task_layout_page(layout_browser: Browser) -> Iterator[Page]:
    stylesheet = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
    page = layout_browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        page.set_content(
            f"<style>{stylesheet}.task-fixture{{width:min(calc(100% - 40px),1000px);margin:20px;}}</style>{TASK_LAYOUT_FIXTURE}"
        )
        yield page
    finally:
        page.close()


def _box(page: Page, selector: str) -> dict[str, float]:
    box = page.locator(selector).bounding_box()
    assert box is not None
    return box


def _assert_no_horizontal_overflow(page: Page) -> None:
    dimensions = page.evaluate(
        "() => ({client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth})"
    )
    assert dimensions["scroll"] <= dimensions["client"]


def test_desktop_workspace_keeps_checks_below_preview_and_editor_to_the_right(
    responsive_page: Page,
) -> None:
    responsive_page.set_viewport_size({"width": 1440, "height": 900})

    stage = _box(responsive_page, ".stage-panel")
    inspector = _box(responsive_page, ".inspector-panel")
    editor = _box(responsive_page, ".inspect-column")

    _assert_no_horizontal_overflow(responsive_page)
    assert stage["width"] > 700
    assert inspector["y"] >= stage["y"] + stage["height"]
    assert inspector["width"] == pytest.approx(stage["width"], abs=1)
    assert editor["x"] >= stage["x"] + stage["width"]


@pytest.mark.parametrize(
    ("width", "height", "minimum_preview_width"),
    [(900, 1000, 700), (390, 844, 300)],
)
def test_narrow_workspace_stacks_preview_checks_then_editor_without_collapse(
    responsive_page: Page,
    width: int,
    height: int,
    minimum_preview_width: int,
) -> None:
    responsive_page.set_viewport_size({"width": width, "height": height})

    preview = _box(responsive_page, ".preview-stage")
    stage = _box(responsive_page, ".stage-panel")
    inspector = _box(responsive_page, ".inspector-panel")
    editor = _box(responsive_page, ".inspect-column")

    _assert_no_horizontal_overflow(responsive_page)
    assert preview["width"] >= minimum_preview_width
    assert inspector["y"] >= stage["y"] + stage["height"]
    assert editor["y"] >= inspector["y"] + inspector["height"]
    assert inspector["width"] == pytest.approx(stage["width"], abs=1)


def test_tablet_header_wraps_before_controls_overflow(responsive_page: Page) -> None:
    responsive_page.set_viewport_size({"width": 900, "height": 1000})

    brand = _box(responsive_page, ".brand")
    actions = _box(responsive_page, ".top-actions")

    _assert_no_horizontal_overflow(responsive_page)
    assert actions["y"] >= brand["y"] + brand["height"]


def test_mobile_object_metadata_has_visible_field_labels(responsive_page: Page) -> None:
    responsive_page.set_viewport_size({"width": 390, "height": 844})

    labels = responsive_page.locator(
        ".object-row:not(.object-row-head) span"
    ).evaluate_all(
        "nodes => nodes.map(node => getComputedStyle(node, '::before').content.replaceAll('\\\"', ''))"
    )

    assert labels == ["Type", "Position", "Size"]


def test_object_renderer_supplies_mobile_metadata_labels() -> None:
    source = (WEB_ROOT / "studio.js").read_text(encoding="utf-8")

    assert 'semantic.dataset.label = "Type";' in source
    assert 'position.dataset.label = "Position";' in source
    assert 'size.dataset.label = "Size";' in source


def test_task_layout_prioritizes_tests_and_collapses_runs_on_desktop(task_layout_page: Page) -> None:
    task_layout_page.set_viewport_size({"width": 1440, "height": 900})

    tests = _box(task_layout_page, ".task-tests")
    runs = _box(task_layout_page, ".task-trajectory-head")
    content = _box(task_layout_page, ".task-card-content")
    actions = _box(task_layout_page, ".task-card-actions")
    head = _box(task_layout_page, ".task-card-head")

    _assert_no_horizontal_overflow(task_layout_page)
    assert actions["y"] < content["y"]
    assert actions["y"] + actions["height"] <= head["y"] + head["height"] + 1
    assert runs["y"] >= tests["y"] + tests["height"]
    assert not task_layout_page.locator(".task-trajectory-list").is_visible()

    task_layout_page.locator(".task-trajectory-head").click()
    replay = _box(task_layout_page, ".task-replay-button")
    assert task_layout_page.locator(".task-trajectory-list").is_visible()
    assert replay["height"] <= 31


def test_task_layout_stacks_cleanly_on_mobile(task_layout_page: Page) -> None:
    task_layout_page.set_viewport_size({"width": 390, "height": 844})
    task_layout_page.locator(".task-trajectory-history").evaluate("element => { element.open = false; }")

    card = _box(task_layout_page, ".task-card")
    tests = _box(task_layout_page, ".task-tests")
    runs = _box(task_layout_page, ".task-trajectory-head")
    actions = _box(task_layout_page, ".task-card-actions")
    content = _box(task_layout_page, ".task-card-content")

    _assert_no_horizontal_overflow(task_layout_page)
    assert actions["y"] < content["y"]
    assert runs["y"] >= tests["y"] + tests["height"]
    assert actions["width"] <= card["width"]
