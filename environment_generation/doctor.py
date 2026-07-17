"""Installation diagnostics for the Environment Generation harness."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 14)
REQUIRED_MODULES = (
    ("mujoco", "MuJoCo"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("playwright.sync_api", "Playwright"),
    ("pydantic", "Pydantic"),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    fix: str | None = None


def check_python(version_info: Sequence[int] | None = None) -> CheckResult:
    version = tuple(version_info or sys.version_info)
    short = version[:3]
    supported = MIN_PYTHON <= short[:2] < MAX_PYTHON_EXCLUSIVE
    detail = f"Python {'.'.join(str(part) for part in short)}"
    return CheckResult(
        name="Python",
        passed=supported,
        detail=detail if supported else f"{detail} is unsupported",
        fix=None if supported else "Install Python 3.10, 3.11, 3.12, or 3.13.",
    )


def check_dependencies(
    importer: Callable[[str], Any] = importlib.import_module,
) -> CheckResult:
    missing: list[str] = []
    for module_name, display_name in REQUIRED_MODULES:
        try:
            importer(module_name)
        except (ImportError, ModuleNotFoundError):
            missing.append(display_name)
    if missing:
        return CheckResult(
            name="Python dependencies",
            passed=False,
            detail=f"Missing: {', '.join(missing)}",
            fix="Run `uv sync`, then retry.",
        )
    return CheckResult(
        name="Python dependencies",
        passed=True,
        detail="MuJoCo, Playwright, and supporting libraries are installed",
    )


def _command_output(completed: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip())


def check_codex(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CheckResult:
    executable = which("codex")
    if executable is None:
        return CheckResult(
            name="Codex CLI",
            passed=False,
            detail="The codex command is not on PATH",
            fix="Install it with npm install -g @openai/codex, then run codex login.",
        )

    try:
        version_result = runner(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="Codex CLI",
            passed=False,
            detail=f"Could not run Codex: {exc}",
            fix="Reinstall the Codex CLI and ensure codex is on PATH.",
        )
    if version_result.returncode != 0:
        detail = _command_output(version_result) or "Codex version check failed"
        return CheckResult(
            name="Codex CLI",
            passed=False,
            detail=detail,
            fix="Reinstall the Codex CLI and ensure codex is on PATH.",
        )

    try:
        login_result = runner(
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="Codex CLI",
            passed=False,
            detail=f"Could not verify Codex authentication: {exc}",
            fix="Run codex login, then retry the setup check.",
        )
    if login_result.returncode != 0:
        detail = _command_output(login_result) or "Codex is not authenticated"
        return CheckResult(
            name="Codex CLI",
            passed=False,
            detail=detail,
            fix="Run codex login, then retry the setup check.",
        )

    version_output = (version_result.stdout or _command_output(version_result)).strip()
    login_output = (login_result.stdout or _command_output(login_result)).strip()
    version = version_output.splitlines()[-1]
    login = login_output.splitlines()[-1]
    return CheckResult(
        name="Codex CLI",
        passed=True,
        detail=f"{version}; {login}",
    )


def check_chromium(
    playwright_factory: Callable[[], Any] | None = None,
) -> CheckResult:
    if playwright_factory is None:
        try:
            from playwright.sync_api import sync_playwright
        except (ImportError, ModuleNotFoundError):
            return CheckResult(
                name="Chromium",
                passed=False,
                detail="Playwright is not installed",
                fix="Run `uv sync`, then retry.",
            )
        playwright_factory = sync_playwright

    try:
        with playwright_factory() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                return CheckResult(
                    name="Chromium",
                    passed=False,
                    detail="The Playwright Chromium runtime is missing",
                    fix="Run `uv run playwright install chromium`, then retry.",
                )
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # Playwright reports driver failures through several exception types.
        detail = f"Could not launch the browser runtime: {exc}"
        if "host system is missing dependencies" in str(exc).lower():
            fix = (
                "Run `uv run playwright install --with-deps chromium`, then retry."
            )
        else:
            fix = "Run `uv run playwright install --force chromium`, then retry."
        return CheckResult(
            name="Chromium",
            passed=False,
            detail=detail,
            fix=fix,
        )
    return CheckResult(
        name="Chromium",
        passed=True,
        detail=f"Browser runtime launched from {executable}",
    )


def check_bundled_assets(package_root: Path | None = None) -> CheckResult:
    root = package_root or Path(__file__).resolve().parent
    required = (
        root / "studio_web" / "index.html",
        root / "studio_web" / "vendor" / "three.module.js",
        root / "studio_web" / "vendor" / "GLTFLoader.js",
        root / "studio_web" / "courtyard_assets.js",
        root / "studio_web" / "visual_renderer.js",
        root / "studio_web" / "assets" / "courtyard" / "SOURCES.md",
        root / "studio_web" / "assets" / "courtyard" / "kenney_platformer" / "crate.glb",
        root
        / "studio_web"
        / "assets"
        / "courtyard"
        / "kenney_platformer"
        / "platform-ramp.glb",
        root
        / "studio_web"
        / "assets"
        / "courtyard"
        / "kenney_nature"
        / "tree_default.glb",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        return CheckResult(
            name="Bundled assets",
            passed=False,
            detail=f"Missing: {', '.join(missing)}",
            fix="Reinstall the project from a complete repository checkout.",
        )
    return CheckResult(
        name="Bundled assets",
        passed=True,
        detail="Studio, Three.js, and courtyard assets are available",
    )


def run_checks(
    *,
    include_codex: bool = True,
    include_browser: bool = True,
) -> tuple[CheckResult, ...]:
    checks = [check_python(), check_dependencies()]
    if include_codex:
        checks.append(check_codex())
    if include_browser:
        checks.append(check_chromium())
    checks.append(check_bundled_assets())
    return tuple(checks)


def _print_human(checks: Sequence[CheckResult], *, quiet: bool) -> None:
    failed = [check for check in checks if not check.passed]
    if quiet and not failed:
        print("Environment Generation is ready.")
        return
    if not quiet:
        print("Environment Generation setup check")
    for check in checks:
        if quiet and check.passed:
            continue
        marker = "OK" if check.passed else "ERROR"
        print(f"[{marker}] {check.name}: {check.detail}")
        if check.fix:
            print(f"        Fix: {check.fix}")
    if failed:
        print(f"\n{len(failed)} setup check(s) need attention.")
    elif not quiet:
        print("\nAll setup checks passed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Environment Generation installation.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    parser.add_argument("--quiet", action="store_true", help="Print only failures or the final success line.")
    parser.add_argument(
        "--skip-codex",
        action="store_true",
        help="Skip the Codex CLI and authentication check.",
    )
    parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="Skip the local Chromium runtime check.",
    )
    args = parser.parse_args(argv)
    checks = run_checks(
        include_codex=not args.skip_codex,
        include_browser=not args.skip_browser,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        _print_human(checks, quiet=args.quiet)
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
