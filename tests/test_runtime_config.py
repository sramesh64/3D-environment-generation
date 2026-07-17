from __future__ import annotations

from environment_generation.runtime_config import (
    rendering_subprocess_env,
    require_runtime_env,
    runtime_env_key,
    runtime_env_value,
)


def test_runtime_value_reads_current_name(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_GENERATION_TASK_MODEL", "current")

    assert runtime_env_key("TASK_MODEL") == "ENVIRONMENT_GENERATION_TASK_MODEL"
    assert runtime_env_value("TASK_MODEL") == "current"


def test_require_runtime_env_reads_current_name(monkeypatch) -> None:
    monkeypatch.delenv("ENVIRONMENT_GENERATION_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("ENVIRONMENT_GENERATION_OUTPUT_ROOT", "/tmp/output")

    assert require_runtime_env("OUTPUT_ROOT") == "/tmp/output"


def test_rendering_subprocess_env_forwards_only_rendering_configuration() -> None:
    env = rendering_subprocess_env(
        {
            "DISPLAY": ":99",
            "XAUTHORITY": "/tmp/xvfb/Xauthority",
            "MUJOCO_GL": "glfw",
            "PYOPENGL_PLATFORM": "glx",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "UNRELATED": "ignored",
            "EMPTY": "",
        }
    )

    assert env == {
        "DISPLAY": ":99",
        "XAUTHORITY": "/tmp/xvfb/Xauthority",
        "MUJOCO_GL": "glfw",
        "PYOPENGL_PLATFORM": "glx",
        "LIBGL_ALWAYS_SOFTWARE": "1",
    }
