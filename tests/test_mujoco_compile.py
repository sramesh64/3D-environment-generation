from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

import pytest

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.mujoco_compile import compile_spec_to_mjcf, validate_mjcf_loads, write_mjcf


def _spec():
    builder = EnvSpec3DBuilder("compile_scene", description="compile")
    builder.make_box_goal_scene()
    builder.add_hazard_zone(2.5, -2.5, id="hazard")
    return builder.finalize()


def test_mjcf_contains_named_objects_and_cameras() -> None:
    xml = compile_spec_to_mjcf(_spec())
    root = ET.fromstring(xml)

    assert root.attrib["model"] == "compile_scene"
    global_visual = root.find("./visual/global")
    assert global_visual is not None
    assert global_visual.attrib["offwidth"] == "1280"
    assert global_visual.attrib["offheight"] == "820"
    assert root.find("./asset/texture[@name='retro_adventure_sky']") is not None
    assert root.find("./asset/texture[@name='retro_adventure_grass']") is not None
    assert root.find("./asset/texture[@name='retro_adventure_hazard']") is not None
    ground_material = root.find("./asset/material[@name='ground']")
    assert ground_material is not None
    assert ground_material.attrib["texture"] == "retro_adventure_grass"
    assert ground_material.attrib["texrepeat"] == "4 3"
    hazard_material = root.find("./asset/material[@name='hazard']")
    assert hazard_material is not None
    assert hazard_material.attrib["texture"] == "retro_adventure_hazard"
    assert hazard_material.attrib["texrepeat"] == "6 4"
    assert root.find(".//light[@name='rim_light']") is not None
    assert root.find(".//camera[@name='overview']") is not None
    assert root.find(".//geom[@name='ground']") is not None
    assert root.find(".//geom[@name='hazard']").attrib["material"] == "hazard"
    assert root.find(".//body[@name='pushable_box']") is not None


def test_mujoco_loads_generated_xml_when_available(tmp_path) -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed")
    xml_path = write_mjcf(_spec(), tmp_path / "world.xml")

    report = validate_mjcf_loads(xml_path)

    assert report["valid"], report
    assert report["ngeom"] >= 5


def test_floor_sensor_visuals_are_thin_without_changing_spec_volume() -> None:
    builder = EnvSpec3DBuilder("sensor_visual", description="tall logical floor sensors")
    builder.add_ground_plane(12, 8)
    builder.add_hazard_zone(0, 0, width=3, depth=2, height=1.4, id="hazard")
    spec = builder.finalize()

    root = ET.fromstring(compile_spec_to_mjcf(spec))
    hazard = root.find(".//geom[@name='hazard']")

    assert hazard is not None
    assert [float(value) for value in hazard.attrib["size"].split()] == pytest.approx([1.5, 1.0, 0.05])
    assert [float(value) for value in hazard.attrib["pos"].split()] == pytest.approx([0.0, 0.0, 0.05])
    assert next(obj for obj in spec.objects if obj.id == "hazard").size[2] == 1.4
