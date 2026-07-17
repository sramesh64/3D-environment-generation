from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from environment_generation.studio_view_context import (
    active_screen_requirements,
    load_studio_view_context,
    persist_studio_view_context,
    submitted_view_image_path,
)


def _view_context() -> dict:
    return {
        "version": 2,
        "capture_kind": "structured_before_edit",
        "captured_at": "2026-07-12T00:00:00Z",
        "screen_space": {
            "camera": {
                "position": [3.4, 11.8, 9.5],
                "target": [0, 0, 0.5],
                "fov_y_degrees": 45,
                "aspect": 16 / 9,
            },
            "regions": {
                "bottom_left": {
                    "bounds_uv": {"left": 0.05, "top": 0.6, "right": 0.34, "bottom": 0.9},
                    "anchor": {"world_position": [-5, 5, 0], "screen_uv": [0.2, 0.75]},
                }
            },
        },
    }


def _png_data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (960, 540), (30, 90, 140)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_persisted_submitted_view_keeps_image_camera_and_prompt_requirement(tmp_path) -> None:
    payload = persist_studio_view_context(
        scene_dir=tmp_path,
        review_id="turn-0002-test",
        prompt="add an agent in the bottom left",
        view_context=_view_context(),
        image_data_url=_png_data_url(),
    )

    image_path = submitted_view_image_path(tmp_path, payload)
    assert image_path == tmp_path / "visual_reviews" / "turn-0002-test" / "submitted_view.png"
    assert image_path.is_file()
    assert payload["submitted_image"]["width"] == 960
    assert payload["requirements"][0]["region"] == "bottom_left"
    assert load_studio_view_context(tmp_path)["review_id"] == "turn-0002-test"
    assert active_screen_requirements(payload, prompt="add an agent in the bottom left")
    assert not active_screen_requirements(payload, prompt="move the box")


def test_submitted_view_rejects_invalid_or_oversized_image_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="PNG data URL"):
        persist_studio_view_context(
            scene_dir=tmp_path,
            review_id="turn-invalid",
            prompt="bottom left",
            view_context=_view_context(),
            image_data_url="data:image/jpeg;base64,aaaa",
        )
