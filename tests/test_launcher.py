from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_generation import launcher
from environment_generation.launcher import (
    LauncherError,
    ensure_chromium,
    run_under_virtual_display,
)


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


def test_headless_linux_reexecs_under_xvfb() -> None:
    calls: list[tuple[str, list[str]]] = []
    messages: list[str] = []

    started = run_under_virtual_display(
        ["--host", "0.0.0.0"],
        platform="linux",
        environ={},
        which=lambda _name: "/usr/bin/xvfb-run",
        execv=lambda executable, args: calls.append((executable, args)),
        announce=messages.append,
    )

    assert started is True
    assert calls == [
        (
            "/usr/bin/xvfb-run",
            [
                "/usr/bin/xvfb-run",
                "-a",
                launcher.sys.executable,
                "-m",
                "environment_generation.launcher",
                "--host",
                "0.0.0.0",
                "--no-open",
            ],
        )
    ]
    assert messages == [
        "Headless Linux detected; starting a virtual display for MuJoCo rendering..."
    ]


@pytest.mark.parametrize(
    "platform,environ",
    [
        ("darwin", {}),
        ("linux", {"DISPLAY": ":0"}),
        ("linux", {"MUJOCO_GL": "egl"}),
        ("linux", {"ENVIRONMENT_GENERATION_MUJOCO_GL": "osmesa"}),
    ],
)
def test_virtual_display_preserves_existing_rendering_configuration(
    platform: str,
    environ: dict[str, str],
) -> None:
    started = run_under_virtual_display(
        [],
        platform=platform,
        environ=environ,
        which=lambda _name: pytest.fail("xvfb should not be inspected"),
        execv=lambda *_args: pytest.fail("process should not be replaced"),
    )

    assert started is False


def test_main_runs_diagnostics_then_forwards_server_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(launcher, "run_under_virtual_display", lambda _args: False)
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
    monkeypatch.setattr(launcher, "run_under_virtual_display", lambda _args: False)
    monkeypatch.setattr(launcher, "ensure_chromium", lambda: False)
    monkeypatch.setattr(launcher, "doctor_main", lambda _args: 1)
    monkeypatch.setattr(
        launcher,
        "studio_main",
        lambda _args: pytest.fail("server should not start"),
    )

    assert launcher.main([]) == 1
