from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}


def _source_files() -> list[Path]:
    return [
        path
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not IGNORED_PARTS.intersection(path.relative_to(REPOSITORY_ROOT).parts)
    ]


def test_retired_product_name_is_absent_from_paths_and_source() -> None:
    retired_names = tuple(
        ("sim" + separator + suffix).casefold()
        for suffix in ("world", "coder")
        for separator in ("", " ", "-", "_")
    )
    matches: list[str] = []

    for path in _source_files():
        relative = path.relative_to(REPOSITORY_ROOT)
        if any(name in str(relative).casefold() for name in retired_names):
            matches.append(f"path: {relative}")
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(name in contents.casefold() for name in retired_names):
            matches.append(f"contents: {relative}")

    assert matches == []


def test_personal_home_paths_are_absent_from_repository_text() -> None:
    home_markers = ("/" + "Users" + "/", "C:" + "\\Users\\")
    matches: list[str] = []

    for path in _source_files():
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in contents for marker in home_markers):
            matches.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert matches == []
