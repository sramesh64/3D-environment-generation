from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from environment_generation.doctor import (
    check_bundled_assets,
    check_chromium,
    check_codex,
    check_dependencies,
    check_python,
    run_checks,
)


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_python_check_accepts_supported_versions_and_rejects_unsupported_versions() -> None:
    assert check_python((3, 10, 0)).passed is True
    assert check_python((3, 13, 9)).passed is True

    result = check_python((3, 14, 0))

    assert result.passed is False
    assert result.fix is not None
    assert "3.13" in result.fix


def test_dependency_check_lists_missing_display_names() -> None:
    def importer(module_name: str) -> object:
        if module_name in {"mujoco", "PIL"}:
            raise ModuleNotFoundError(module_name)
        return object()

    result = check_dependencies(importer)

    assert result.passed is False
    assert result.detail == "Missing: MuJoCo, Pillow"
    assert result.fix is not None


def test_codex_check_explains_missing_cli() -> None:
    result = check_codex(which=lambda _name: None)

    assert result.passed is False
    assert "not on PATH" in result.detail
    assert result.fix == "Install it with npm install -g @openai/codex, then run codex login."


def test_codex_check_requires_authentication() -> None:
    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-2:] == ["login", "status"]:
            return _completed(args, returncode=1, stderr="Not logged in")
        return _completed(args, stdout="codex-cli 1.2.3")

    result = check_codex(which=lambda _name: "/usr/local/bin/codex", runner=runner)

    assert result.passed is False
    assert result.detail == "Not logged in"
    assert result.fix == "Run codex login, then retry the setup check."


def test_codex_check_reports_version_and_login_status() -> None:
    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-2:] == ["login", "status"]:
            return _completed(args, stdout="Logged in using ChatGPT")
        return _completed(args, stdout="codex-cli 1.2.3")

    result = check_codex(which=lambda _name: "/usr/local/bin/codex", runner=runner)

    assert result.passed is True
    assert result.detail == "codex-cli 1.2.3; Logged in using ChatGPT"


def test_chromium_check_uses_playwright_executable(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    executable.touch()

    class PlaywrightContext:
        def __enter__(self) -> object:
            browser = SimpleNamespace(close=lambda: None)
            return SimpleNamespace(
                chromium=SimpleNamespace(
                    executable_path=str(executable),
                    launch=lambda **_kwargs: browser,
                )
            )

        def __exit__(self, *_args: object) -> None:
            return None

    result = check_chromium(lambda: PlaywrightContext())

    assert result.passed is True
    assert str(executable) in result.detail


def test_chromium_check_reports_missing_runtime(tmp_path: Path) -> None:
    class PlaywrightContext:
        def __enter__(self) -> object:
            return SimpleNamespace(
                chromium=SimpleNamespace(
                    executable_path=str(tmp_path / "missing"),
                    launch=lambda **_kwargs: None,
                )
            )

        def __exit__(self, *_args: object) -> None:
            return None

    result = check_chromium(lambda: PlaywrightContext())

    assert result.passed is False
    assert "missing" in result.detail.lower()


def test_chromium_check_reports_linux_host_dependency_fix(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    executable.touch()

    class PlaywrightContext:
        def __enter__(self) -> object:
            def fail_to_launch(**_kwargs: object) -> None:
                raise RuntimeError("Host system is missing dependencies to run browsers")

            return SimpleNamespace(
                chromium=SimpleNamespace(
                    executable_path=str(executable),
                    launch=fail_to_launch,
                )
            )

        def __exit__(self, *_args: object) -> None:
            return None

    result = check_chromium(lambda: PlaywrightContext())

    assert result.passed is False
    assert result.fix == (
        "Run `uv run playwright install --with-deps chromium`, then retry."
    )


def test_bundled_asset_check_matches_repository_layout() -> None:
    assert check_bundled_assets().passed is True


def test_run_checks_can_skip_external_cli_and_browser() -> None:
    checks = run_checks(include_codex=False, include_browser=False)

    assert [check.name for check in checks] == [
        "Python",
        "Python dependencies",
        "Bundled assets",
    ]
