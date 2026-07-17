from __future__ import annotations

import importlib.util

import pytest

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.mujoco_compile import write_mjcf
from environment_generation.preview import image_nonblank, render_orbit_previews, render_preview


def test_preview_renders_nonblank_image_when_dependencies_available(tmp_path) -> None:
    if importlib.util.find_spec("mujoco") is None or importlib.util.find_spec("PIL") is None:
        pytest.skip("mujoco/Pillow preview dependencies are not installed")
    builder = EnvSpec3DBuilder("preview_scene", description="preview")
    builder.make_box_goal_scene()
    xml_path = write_mjcf(builder.finalize(), tmp_path / "world.xml")
    image_path = render_preview(xml_path, tmp_path / "preview.png", camera="overview", width=320, height=220)

    assert image_path.is_file()
    assert image_nonblank(image_path)


def test_orbit_previews_render_nonblank_frames_when_dependencies_available(tmp_path) -> None:
    if importlib.util.find_spec("mujoco") is None or importlib.util.find_spec("PIL") is None:
        pytest.skip("mujoco/Pillow preview dependencies are not installed")
    builder = EnvSpec3DBuilder("orbit_scene", description="orbit")
    builder.make_box_goal_scene()
    xml_path = write_mjcf(builder.finalize(), tmp_path / "world.xml")
    paths = [tmp_path / f"preview_orbit_{index:02d}.png" for index in range(4)]

    image_paths = render_orbit_previews(xml_path, paths, width=320, height=220)

    assert image_paths == paths
    assert all(path.is_file() for path in image_paths)
    assert all(image_nonblank(path) for path in image_paths)
