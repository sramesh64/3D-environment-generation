from __future__ import annotations

import json
from types import SimpleNamespace

from environment_generation.codex_models import (
    CodexModelCatalog,
    load_configured_codex_model,
    model_display_name,
    normalize_model_catalog,
)


def test_model_catalog_keeps_only_visible_api_models_and_sorts_by_priority() -> None:
    models = normalize_model_catalog(
        {
            "models": [
                {
                    "slug": "slower",
                    "display_name": "Slower",
                    "description": "Second model",
                    "visibility": "list",
                    "supported_in_api": True,
                    "priority": 20,
                    "default_reasoning_level": "high",
                    "supported_reasoning_levels": [{"effort": "medium"}, {"effort": "high"}],
                },
                {
                    "slug": "first",
                    "display_name": "First",
                    "description": "First model",
                    "visibility": "list",
                    "supported_in_api": True,
                    "priority": 1,
                },
                {"slug": "hidden", "visibility": "hide", "supported_in_api": True},
                {"slug": "unsupported", "visibility": "list", "supported_in_api": False},
                {"slug": "first", "display_name": "Duplicate", "visibility": "list"},
            ]
        }
    )

    assert [model["id"] for model in models] == ["first", "slower"]
    assert models[1]["reasoning_levels"] == ["medium", "high"]
    assert models[1]["default_reasoning_level"] == "high"


def test_model_catalog_is_cached_and_returns_independent_values() -> None:
    calls = []
    now = [100.0]

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "models": [
                        {
                            "slug": "test-model",
                            "display_name": "Test Model",
                            "visibility": "list",
                            "supported_in_api": True,
                            "priority": 0,
                        }
                    ]
                }
            ),
        )

    catalog = CodexModelCatalog(
        ttl_seconds=10,
        runner=runner,
        clock=lambda: now[0],
        configured_model_loader=lambda: "",
    )
    first = catalog.get()
    first["models"].clear()
    second = catalog.get()

    assert second["models"][0]["id"] == "test-model"
    assert len(calls) == 1
    assert calls[0][0] == ["codex", "debug", "models"]
    assert calls[0][1]["timeout"] == 15

    now[0] += 11
    catalog.get()
    assert len(calls) == 2


def test_model_catalog_falls_back_to_codex_default_when_discovery_fails() -> None:
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    result = CodexModelCatalog(runner=runner, configured_model_loader=lambda: "").get()

    assert result["status"] == "unavailable"
    assert result["models"] == []
    assert "default" in result["message"].lower()


def test_configured_model_is_merged_when_older_cli_catalog_omits_it() -> None:
    def runner(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-5.5",
                            "display_name": "GPT-5.5",
                            "visibility": "list",
                            "supported_in_api": True,
                            "priority": 0,
                        }
                    ]
                }
            ),
        )

    result = CodexModelCatalog(
        runner=runner,
        configured_model_loader=lambda: "gpt-5.6-sol",
    ).get()

    assert [model["id"] for model in result["models"]] == ["gpt-5.6-sol", "gpt-5.5"]
    assert result["models"][0]["name"] == "GPT-5.6 Sol"
    assert result["default_model"] == {
        "id": "gpt-5.6-sol",
        "name": "GPT-5.6 Sol",
        "source": "user_config",
    }


def test_configured_model_uses_catalog_metadata_when_listed() -> None:
    def runner(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-5.6-sol",
                            "display_name": "GPT-5.6 Sol (catalog)",
                            "visibility": "list",
                            "supported_in_api": True,
                            "priority": 0,
                        }
                    ]
                }
            ),
        )

    result = CodexModelCatalog(
        runner=runner,
        configured_model_loader=lambda: "gpt-5.6-sol",
    ).get()

    assert len(result["models"]) == 1
    assert result["default_model"]["name"] == "GPT-5.6 Sol (catalog)"


def test_configured_model_remains_available_when_catalog_discovery_fails() -> None:
    def runner(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    result = CodexModelCatalog(
        runner=runner,
        configured_model_loader=lambda: "gpt-5.6-sol",
    ).get()

    assert result["status"] == "ready"
    assert [model["id"] for model in result["models"]] == ["gpt-5.6-sol"]
    assert "unavailable" in result["message"].lower()


def test_configured_model_change_invalidates_cached_catalog() -> None:
    calls = []
    configured_model = ["gpt-5.5"]

    def runner(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-5.5",
                            "display_name": "GPT-5.5",
                            "visibility": "list",
                            "supported_in_api": True,
                            "priority": 0,
                        }
                    ]
                }
            ),
        )

    catalog = CodexModelCatalog(
        runner=runner,
        configured_model_loader=lambda: configured_model[0],
    )
    catalog.get()
    configured_model[0] = "gpt-5.6-sol"
    result = catalog.get()

    assert len(calls) == 2
    assert result["default_model"]["id"] == "gpt-5.6-sol"


def test_load_configured_codex_model_reads_root_model(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-5.6-sol"\n[features]\nexample = true\n', encoding="utf-8")

    assert load_configured_codex_model(config_path) == "gpt-5.6-sol"
    assert load_configured_codex_model(tmp_path / "missing.toml") == ""
    assert model_display_name("gpt-5.6-sol") == "GPT-5.6 Sol"
