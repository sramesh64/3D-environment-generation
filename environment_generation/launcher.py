"""Cross-platform launcher for the uv-managed application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .doctor import main as doctor_main
from .studio_server import main as studio_main


class LauncherError(RuntimeError):
    """Raised when a first-run dependency cannot be prepared."""


def chromium_executable(
    playwright_factory: Callable[[], Any] | None = None,
) -> Path:
    if playwright_factory is None:
        try:
            from playwright.sync_api import sync_playwright
        except (ImportError, ModuleNotFoundError) as exc:
            raise LauncherError(
                "Playwright is unavailable. Run `uv sync` and try again."
            ) from exc
        playwright_factory = sync_playwright

    try:
        with playwright_factory() as playwright:
            return Path(playwright.chromium.executable_path)
    except Exception as exc:
        raise LauncherError(f"Could not inspect the Chromium runtime: {exc}") from exc


def ensure_chromium(
    *,
    playwright_factory: Callable[[], Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    announce: Callable[[str], None] = print,
) -> bool:
    """Install the Playwright browser once and return whether installation ran."""

    executable = chromium_executable(playwright_factory)
    if executable.is_file():
        return False

    announce("First run: installing the local Chromium runtime...")
    command = [sys.executable, "-m", "playwright", "install", "chromium"]
    try:
        completed = runner(command, check=False)
    except OSError as exc:
        raise LauncherError(f"Could not start the Chromium installer: {exc}") from exc
    if completed.returncode != 0:
        raise LauncherError(
            "Chromium installation failed. Run `uv run playwright install chromium` "
            "to see the installer output and retry."
        )

    executable = chromium_executable(playwright_factory)
    if not executable.is_file():
        raise LauncherError(
            "Chromium installation completed without creating the expected runtime. "
            "Run `uv run playwright install --force chromium` and retry."
        )
    announce("Chromium is ready.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    try:
        ensure_chromium()
    except LauncherError as exc:
        print(f"Environment Generation could not start:\n{exc}", file=sys.stderr)
        return 1

    if doctor_main(["--quiet"]) != 0:
        return 1

    studio_main(list(argv) if argv is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

