from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


MODEL_CATALOG_TTL_SECONDS = 5 * 60
MODEL_CATALOG_TIMEOUT_SECONDS = 15


def codex_config_path() -> Path:
    codex_home = str(os.environ.get("CODEX_HOME") or "").strip()
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def load_configured_codex_model(config_path: Path | None = None) -> str:
    path = config_path or codex_config_path()
    try:
        with path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return ""
    return str(config.get("model") or "").strip()


def model_display_name(model_id: str) -> str:
    normalized = model_id.strip()
    if not normalized:
        return ""
    if normalized.startswith("gpt-"):
        parts = normalized[4:].split("-")
        version = parts.pop(0)
        suffix = " ".join(part.title() for part in parts)
        return f"GPT-{version}{f' {suffix}' if suffix else ''}"
    return normalized


def normalize_model_catalog(value: Any) -> list[dict[str, Any]]:
    raw_models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(raw_models, list):
        raise ValueError("Codex returned an invalid model catalog")

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_models:
        if not isinstance(raw, dict) or str(raw.get("visibility") or "") != "list":
            continue
        if raw.get("supported_in_api") is False:
            continue
        model_id = str(raw.get("slug") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        try:
            priority = int(raw.get("priority", 1000))
        except (TypeError, ValueError):
            priority = 1000
        reasoning_levels = [
            str(level.get("effort") or "").strip()
            for level in raw.get("supported_reasoning_levels") or []
            if isinstance(level, dict) and str(level.get("effort") or "").strip()
        ]
        models.append(
            {
                "id": model_id,
                "name": str(raw.get("display_name") or model_id).strip() or model_id,
                "description": str(raw.get("description") or "").strip(),
                "priority": priority,
                "default_reasoning_level": str(raw.get("default_reasoning_level") or "").strip(),
                "reasoning_levels": reasoning_levels,
            }
        )
    models.sort(key=lambda model: (model["priority"], model["name"].lower(), model["id"]))
    return models


class CodexModelCatalog:
    def __init__(
        self,
        *,
        ttl_seconds: float = MODEL_CATALOG_TTL_SECONDS,
        timeout_seconds: float = MODEL_CATALOG_TIMEOUT_SECONDS,
        runner: Callable[..., Any] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        configured_model_loader: Callable[[], str] = load_configured_codex_model,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._runner = runner
        self._clock = clock
        self._configured_model_loader = configured_model_loader
        self._lock = threading.Lock()
        self._cached_at = float("-inf")
        self._cached: dict[str, Any] | None = None

    def get(self, *, refresh: bool = False) -> dict[str, Any]:
        now = self._clock()
        try:
            configured_model_id = str(self._configured_model_loader() or "").strip()
        except (OSError, ValueError):
            configured_model_id = ""
        with self._lock:
            cached_default = (self._cached or {}).get("default_model") or {}
            cached_default_id = str(cached_default.get("id") or "")
            if (
                not refresh
                and self._cached is not None
                and cached_default_id == configured_model_id
                and now - self._cached_at < self._ttl_seconds
            ):
                return copy.deepcopy(self._cached)

            catalog = self._load(configured_model_id=configured_model_id)
            self._cached = catalog
            self._cached_at = now
            return copy.deepcopy(catalog)

    def _load(self, *, configured_model_id: str) -> dict[str, Any]:
        discovery_error = ""
        try:
            result = self._runner(
                ["codex", "debug", "models"],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError("Codex model discovery failed")
            models = normalize_model_catalog(json.loads(result.stdout))
            if not models:
                raise RuntimeError("Codex did not report any available models")
        except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, subprocess.TimeoutExpired, ValueError):
            models = []
            discovery_error = "The Codex model catalog is unavailable."

        default_model: dict[str, str] | None = None
        if configured_model_id:
            configured_model = next((model for model in models if model["id"] == configured_model_id), None)
            if configured_model is None:
                configured_model = {
                    "id": configured_model_id,
                    "name": model_display_name(configured_model_id),
                    "description": "Configured as the Codex default in config.toml.",
                    "priority": -1,
                    "default_reasoning_level": "",
                    "reasoning_levels": [],
                }
                models.insert(0, configured_model)
            default_model = {
                "id": configured_model_id,
                "name": str(configured_model.get("name") or model_display_name(configured_model_id)),
                "source": "user_config",
            }

        if not models:
            return {
                "status": "unavailable",
                "source": "codex_cli",
                "models": [],
                "default_model": default_model,
                "message": "The model catalog is unavailable. Codex default remains usable.",
            }
        return {
            "status": "ready",
            "source": "codex_cli_and_config" if configured_model_id else "codex_cli",
            "models": models,
            "default_model": default_model,
            "message": discovery_error,
        }


CODEX_MODEL_CATALOG = CodexModelCatalog()
