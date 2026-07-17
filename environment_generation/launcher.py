"""Cross-platform launcher for the uv-managed application."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .doctor import main as doctor_main
from .studio_server import main as studio_main


class LauncherError(RuntimeError):
    """Raised when a first-run dependency cannot be prepared."""


def run_under_virtual_display(
    argv: Sequence[str],
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    execv: Callable[[str, list[str]], Any] = os.execv,
    announce: Callable[[str], None] = print,
) -> bool:
    """Re-exec headless Linux under Xvfb so MuJoCo can render offscreen."""

    current_platform = platform or sys.platform
    current_env = environ if environ is not None else os.environ
    if (
        not current_platform.startswith("linux")
        or current_env.get("DISPLAY")
        or current_env.get("MUJOCO_GL")
        or current_env.get("ENVIRONMENT_GENERATION_MUJOCO_GL")
    ):
        return False

    xvfb = which("xvfb-run")
    if not xvfb:
        return False

    forwarded = list(argv)
    if "--no-open" not in forwarded:
        forwarded.append("--no-open")
    command = [
        xvfb,
        "-a",
        sys.executable,
        "-m",
        "environment_generation.launcher",
        *forwarded,
    ]
    announce("Headless Linux detected; starting a virtual display for MuJoCo rendering...")
    try:
        execv(xvfb, command)
    except OSError as exc:
        raise LauncherError(f"Could not start the Xvfb virtual display: {exc}") from exc
    return True


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
    forwarded = list(argv) if argv is not None else list(sys.argv[1:])
    try:
        if run_under_virtual_display(forwarded):
            return 0
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
