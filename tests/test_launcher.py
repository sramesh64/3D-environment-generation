from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_generation import launcher
from environment_generation.launcher import LauncherError, ensure_chromium


class _PlaywrightContext:
    def __init__(self, executable: Path) -> None:
        self._executable = executable

    def __enter__(self) -> object:
        return SimpleNamespace(
            chromium=SimpleNamespace(executable_path=str(self._executable))
        )

    def __exit__(self, *_args: object) -> None:
        return None


def test_existing_chromium_skips_installation(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    executable.touch()

    def unexpected_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("installer should not run")

    installed = ensure_chromium(
        playwright_factory=lambda: _PlaywrightContext(executable),
        runner=unexpected_runner,
    )

    assert installed is False


def test_missing_chromium_is_installed_once(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    commands: list[list[str]] = []
    messages: list[str] = []

    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        executable.touch()
        return subprocess.CompletedProcess(args, returncode=0)

    installed = ensure_chromium(
        playwright_factory=lambda: _PlaywrightContext(executable),
        runner=runner,
        announce=messages.append,
    )

    assert installed is True
    assert commands == [
        [launcher.sys.executable, "-m", "playwright", "install", "chromium"]
    ]
    assert messages == [
        "First run: installing the local Chromium runtime...",
        "Chromium is ready.",
    ]


def test_failed_chromium_install_has_actionable_error(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"

    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode=1)

    with pytest.raises(LauncherError, match="uv run playwright install chromium"):
        ensure_chromium(
            playwright_factory=lambda: _PlaywrightContext(executable),
            runner=runner,
        )


def test_main_runs_diagnostics_then_forwards_server_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(launcher, "ensure_chromium", lambda: False)
    monkeypatch.setattr(
        launcher,
        "doctor_main",
        lambda args: calls.append(("doctor", args)) or 0,
    )
    monkeypatch.setattr(
        launcher,
        "studio_main",
        lambda args: calls.append(("studio", args)),
    )

    result = launcher.main(["--port", "3040", "--no-open"])

    assert result == 0
    assert calls == [
        ("doctor", ["--quiet"]),
        ("studio", ["--port", "3040", "--no-open"]),
    ]


def test_main_does_not_start_server_when_diagnostics_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "ensure_chromium", lambda: False)
    monkeypatch.setattr(launcher, "doctor_main", lambda _args: 1)
    monkeypatch.setattr(
        launcher,
        "studio_main",
        lambda _args: pytest.fail("server should not start"),
    )

    assert launcher.main([]) == 1

