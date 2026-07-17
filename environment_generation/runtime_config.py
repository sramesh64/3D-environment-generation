"""Runtime configuration names for Environment Generation."""

from __future__ import annotations

import os
from collections.abc import Mapping


ENVIRONMENT_PREFIX = "ENVIRONMENT_GENERATION_"
RENDERING_ENV_KEYS = (
    "DISPLAY",
    "XAUTHORITY",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE",
)


def runtime_env_key(suffix: str) -> str:
    return f"{ENVIRONMENT_PREFIX}{suffix}"


def runtime_env_value(suffix: str, default: str | None = None) -> str | None:
    return os.getenv(runtime_env_key(suffix)) or default


def require_runtime_env(suffix: str) -> str:
    value = runtime_env_value(suffix)
    if value is None:
        raise RuntimeError(f"{runtime_env_key(suffix)} is required")
    return value


def rendering_subprocess_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the rendering variables that strict MCP subprocesses must inherit."""

    source = environ if environ is not None else os.environ
    return {key: source[key] for key in RENDERING_ENV_KEYS if source.get(key)}


def configure_mujoco_gl() -> None:
    if "MUJOCO_GL" in os.environ:
        return
    configured = runtime_env_value("MUJOCO_GL")
    if configured:
        os.environ["MUJOCO_GL"] = configured
