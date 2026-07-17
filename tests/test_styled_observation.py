from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageStat
import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.styled_observation import StyledObservationRenderer


@pytest.mark.skipif(
    os.getenv("ENVIRONMENT_GENERATION_RUN_STYLED_CAPTURE_TESTS") != "1",
    reason="set ENVIRONMENT_GENERATION_RUN_STYLED_CAPTURE_TESTS=1 to launch local Chromium",
)
def test_styled_observation_renders_nonblank_first_person_frame(tmp_path: Path) -> None:
    builder = EnvSpec3DBuilder("styled_capture", description="styled capture")
    builder.add_ground_plane(width=10, depth=7)
    builder.add_agent_spawn(-2.5, 0, id="robot")
    builder.add_hazard_zone(-0.5, 0.5, width=1.4, depth=1.4, id="hazard")
    builder.add_goal_zone(2.5, 0, width=1.4, depth=1.4, id="goal")
    scene_dir = tmp_path / "styled_capture"
    persist_artifacts(
        spec=builder.finalize(),
        scene_dir=scene_dir,
        trace_records=[],
        render=False,
    )
    visual_scene = json.loads((scene_dir / "visual_scene.json").read_text(encoding="utf-8"))
    agent = next(
        item for item in visual_scene["objects"] if item.get("semantic_type") == "agent"
    )
    output = tmp_path / "styled.png"
    renderer = StyledObservationRenderer(
        visual_scene_path=scene_dir / "visual_scene.json",
        width=640,
        height=360,
    )
    try:
        stats = renderer.render(
            output_path=output,
            objects=[],
            mechanisms=[],
            camera={
                "position": [agent["position"][0], agent["position"][1], 0.86],
                "target": [agent["position"][0] + 1, agent["position"][1], 0.86],
                "fov_y_degrees": 64,
            },
            hidden_source_ids=[agent["source_id"]],
        )
    finally:
        renderer.close()

    with Image.open(output) as image:
        assert image.size == (640, 360)
        assert max(ImageStat.Stat(image.convert("RGB")).var) > 100
    assert stats["colorRange"] >= 8
