"""Headless Three.js capture for task-agent visual observations."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any


STYLED_OBSERVATION_RENDERER = "threejs_styled"
MUJOCO_OBSERVATION_RENDERER = "mujoco_fallback"


class StyledObservationUnavailable(RuntimeError):
    """Raised when the local styled renderer cannot produce a frame."""


class _QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        return


class _StaticHost:
    def __init__(self, root: Path) -> None:
        handler = partial(_QuietStaticHandler, directory=str(root))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class StyledObservationRenderer:
    """Keep one headless browser alive and capture the shared Three.js scene."""

    def __init__(
        self,
        *,
        visual_scene_path: Path,
        width: int = 640,
        height: int = 360,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.web_root = Path(__file__).resolve().parent / "studio_web"
        try:
            visual_scene = json.loads(visual_scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StyledObservationUnavailable(
                f"styled visual scene is unavailable: {visual_scene_path}"
            ) from exc
        if not isinstance(visual_scene, dict):
            raise StyledObservationUnavailable("styled visual scene must be a JSON object")

        self._host: _StaticHost | None = None
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        try:
            from playwright.sync_api import sync_playwright

            self._host = _StaticHost(self.web_root)
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
            )
            self._context = self._browser.new_context(
                viewport={"width": self.width, "height": self.height},
                device_scale_factor=1,
            )
            self._page = self._context.new_page()
            self._page.goto(
                f"{self._host.url}/styled_observation_capture.html",
                wait_until="networkidle",
                timeout=30_000,
            )
            self._page.wait_for_function(
                "() => Boolean(window.environmentGenerationStyledObservation)", timeout=15_000
            )
            self._page.evaluate(
                "scene => window.environmentGenerationStyledObservation.setScene(scene)",
                visual_scene,
            )
        except Exception as exc:
            self.close()
            raise StyledObservationUnavailable(
                "could not start the styled Three.js observation renderer; "
                "install Playwright and its Chromium runtime"
            ) from exc

    def render(
        self,
        *,
        output_path: Path,
        objects: list[dict[str, Any]],
        mechanisms: list[dict[str, Any]],
        camera: dict[str, Any],
        hidden_source_ids: list[str],
    ) -> dict[str, Any]:
        if self._page is None:
            raise StyledObservationUnavailable("styled observation renderer is closed")
        payload = {
            "objects": objects,
            "mechanisms": mechanisms,
            "camera": camera,
            "hidden_source_ids": hidden_source_ids,
        }
        try:
            stats = self._page.evaluate(
                "value => window.environmentGenerationStyledObservation.render(value)", payload
            )
            if not isinstance(stats, dict) or int(stats.get("colorRange") or 0) < 8:
                raise StyledObservationUnavailable("styled observation frame was blank")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._page.locator("#captureStage canvas").screenshot(
                path=str(output_path),
                type="png",
            )
            return stats
        except StyledObservationUnavailable:
            raise
        except Exception as exc:
            raise StyledObservationUnavailable(
                f"styled observation capture failed: {exc}"
            ) from exc

    def close(self) -> None:
        for resource_name in ("_context", "_browser"):
            resource = getattr(self, resource_name, None)
            setattr(self, resource_name, None)
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        playwright = self._playwright
        self._playwright = None
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
        host = self._host
        self._host = None
        if host is not None:
            host.close()
        self._page = None
