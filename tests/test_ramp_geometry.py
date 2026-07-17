from __future__ import annotations

import importlib.util
import math
import xml.etree.ElementTree as ET

import pytest

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.mujoco_compile import compile_spec_to_mjcf
from environment_generation.operations import execute_operation
from environment_generation.ramp_geometry import ramp_geometry_from_object
from environment_generation.schema import SpecValidationError, parse_env_spec_3d


def _ramp_builder(*, yaw: float = 0) -> EnvSpec3DBuilder:
    builder = EnvSpec3DBuilder("ramp_contract", description="ramp geometry contract")
    builder.add_ramp(
        0,
        1,
        z=0,
        length=3,
        width=2,
        rise=1,
        thickness=0.25,
        yaw=yaw,
        id="ramp",
    )
    return builder


def test_ramp_uses_explicit_walkable_endpoints_and_separate_thickness() -> None:
    ramp = _ramp_builder().finalize().objects[0]
    geometry = ramp_geometry_from_object(ramp.model_dump(mode="json"))

    assert geometry.low_end == pytest.approx((0, 1, 0))
    assert geometry.high_end == pytest.approx((3, 1, 1))
    assert geometry.slope_length == pytest.approx(math.sqrt(10))
    assert ramp.size == pytest.approx([3, 2, 0.25])
    assert ramp.metadata == {
        "geometry_version": 2,
        "rise": 1.0,
        "low_end": [0.0, 1.0, 0.0],
    }


def test_set_ramp_geometry_changes_rise_without_thickening_collider() -> None:
    builder = _ramp_builder()

    result = execute_operation(
        builder,
        {"op": "set_ramp_geometry", "args": {"id": "ramp", "rise": 1.5, "length": 4}},
    )

    assert result["success"] is True
    ramp = builder.finalize().objects[0]
    geometry = ramp_geometry_from_object(ramp.model_dump(mode="json"))
    assert geometry.high_end == pytest.approx((4, 1, 1.5))
    assert geometry.thickness == pytest.approx(0.25)
    assert ramp.size == pytest.approx([4, 2, 0.25])


@pytest.mark.parametrize("op,args", [
    ("move_object", {"x": 1}),
    ("rotate_object", {"yaw": 1}),
    ("resize_object", {"height": 1}),
])
def test_generic_transform_operations_direct_ramps_to_typed_operation(op: str, args: dict) -> None:
    result = execute_operation(_ramp_builder(), {"op": op, "args": {"id": "ramp", **args}})

    assert result["success"] is False
    assert "set_ramp_geometry" in result["error"]


def test_legacy_add_ramp_height_argument_is_normalized_to_rise() -> None:
    builder = EnvSpec3DBuilder("legacy_ramp")

    result = execute_operation(
        builder,
        {"op": "add_ramp", "args": {"x": 0, "y": 0, "height": 1.2}},
    )

    assert result["success"] is True
    assert result["operation"]["args"]["rise"] == pytest.approx(1.2)
    assert "height" not in result["operation"]["args"]


def test_canonical_ramp_rejects_inconsistent_manual_position() -> None:
    spec = _ramp_builder().finalize().model_dump(mode="json")
    spec["objects"][0]["position"][0] += 1

    with pytest.raises(SpecValidationError, match="inconsistent"):
        parse_env_spec_3d(spec)


@pytest.mark.parametrize("yaw", [0, math.pi / 2, math.pi, -math.pi / 2])
def test_mjcf_ramp_uses_sloped_length_and_exact_top_endpoints_at_any_yaw(yaw: float) -> None:
    spec = _ramp_builder(yaw=yaw).finalize()
    xml = compile_spec_to_mjcf(spec)
    geom = ET.fromstring(xml).find(".//geom[@name='ramp']")

    assert geom is not None
    assert [float(value) for value in geom.attrib["size"].split()] == pytest.approx(
        [math.sqrt(10) / 2, 1, 0.125]
    )
    angle = math.atan2(1, 3)
    assert [float(value) for value in geom.attrib["xyaxes"].split()] == pytest.approx(
        [
            math.cos(yaw) * math.cos(angle),
            math.sin(yaw) * math.cos(angle),
            math.sin(angle),
            -math.sin(yaw),
            math.cos(yaw),
            0,
        ],
        abs=1e-6,
    )

    if importlib.util.find_spec("mujoco") is None:
        return
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp")
    center = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_slope = math.sqrt(10) / 2
    low = center + rotation @ [-half_slope, 0, 0.125]
    high = center + rotation @ [half_slope, 0, 0.125]
    assert low == pytest.approx([0, 1, 0], abs=2e-6)
    assert high == pytest.approx([3 * math.cos(yaw), 1 + 3 * math.sin(yaw), 1], abs=2e-6)
