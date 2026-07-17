from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .ramp_geometry import ramp_geometry_from_object
from .schema import EnvSpec3D, parse_env_spec_3d


class MuJoCoCompileError(ValueError):
    """Raised when a 3D spec cannot be compiled to MJCF."""


TEXTURES: dict[str, dict[str, str]] = {
    "retro_adventure_sky": {
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "0.55 0.78 1",
        "rgb2": "0.95 0.98 0.72",
        "width": "512",
        "height": "512",
    },
    "retro_adventure_grass": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.62 0.86 0.5",
        "rgb2": "0.43 0.73 0.38",
        "width": "16",
        "height": "16",
    },
    "retro_adventure_wall": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.2 0.5 0.48",
        "rgb2": "0.11 0.32 0.34",
        "width": "12",
        "height": "12",
    },
    "retro_adventure_path": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.86 0.68 0.38",
        "rgb2": "0.65 0.46 0.24",
        "width": "10",
        "height": "10",
    },
    "retro_adventure_warm": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.95 0.5 0.2",
        "rgb2": "0.74 0.24 0.12",
        "width": "8",
        "height": "8",
    },
    "retro_adventure_player": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.18 0.48 1",
        "rgb2": "0.05 0.18 0.68",
        "width": "8",
        "height": "8",
    },
    "retro_adventure_hazard": {
        "type": "2d",
        "builtin": "checker",
        "rgb1": "1 0.72 0.05",
        "rgb2": "0.04 0.05 0.06",
        "width": "16",
        "height": "16",
    },
}

MATERIALS: dict[str, dict[str, str]] = {
    "ground": {
        "rgba": "#8BD66C",
        "texture": "retro_adventure_grass",
        "texrepeat": "4 3",
        "texuniform": "true",
        "reflectance": "0.12",
        "specular": "0.18",
        "shininess": "0.25",
    },
    "wall": {
        "rgba": "#28746F",
        "texture": "retro_adventure_wall",
        "texrepeat": "4 1",
        "texuniform": "true",
        "reflectance": "0.08",
        "specular": "0.18",
        "shininess": "0.2",
    },
    "platform": {
        "rgba": "#C6974C",
        "texture": "retro_adventure_path",
        "texrepeat": "2 2",
        "texuniform": "true",
        "reflectance": "0.1",
        "specular": "0.22",
        "shininess": "0.35",
    },
    "ramp": {
        "rgba": "#C6974C",
        "texture": "retro_adventure_path",
        "texrepeat": "2 1",
        "texuniform": "true",
        "reflectance": "0.1",
        "specular": "0.2",
        "shininess": "0.3",
    },
    "pushable_box": {
        "rgba": "#EF6430",
        "texture": "retro_adventure_warm",
        "texrepeat": "1 1",
        "texuniform": "true",
        "reflectance": "0.08",
        "specular": "0.28",
        "shininess": "0.45",
    },
    "ball": {
        "rgba": "#F0C84A",
        "texture": "retro_adventure_warm",
        "texrepeat": "1 1",
        "texuniform": "true",
        "reflectance": "0.1",
        "specular": "0.35",
        "shininess": "0.5",
    },
    "cylinder": {
        "rgba": "#DF7030",
        "texture": "retro_adventure_warm",
        "texrepeat": "1 1",
        "texuniform": "true",
        "reflectance": "0.08",
        "specular": "0.24",
        "shininess": "0.35",
    },
    "agent": {
        "rgba": "#2D73FF",
        "texture": "retro_adventure_player",
        "texrepeat": "1 1",
        "texuniform": "true",
        "reflectance": "0.1",
        "specular": "0.4",
        "shininess": "0.6",
    },
    "goal": {
        "rgba": "#57F28799",
        "reflectance": "0.03",
        "specular": "0.15",
        "shininess": "0.25",
        "emission": "0.12",
    },
    "target_region": {
        "rgba": "#70D6E866",
        "reflectance": "0.02",
        "specular": "0.16",
        "shininess": "0.3",
        "emission": "0.08",
    },
    "hazard": {
        "rgba": "#FFD13BCC",
        "texture": "retro_adventure_hazard",
        "texrepeat": "6 4",
        "texuniform": "true",
        "reflectance": "0.03",
        "specular": "0.18",
        "shininess": "0.35",
        "emission": "0.1",
    },
    "floor_switch": {
        "rgba": "#49C6E5AA",
        "reflectance": "0.05",
        "specular": "0.3",
        "shininess": "0.55",
        "emission": "0.2",
    },
    "gate": {
        "rgba": "#73806F",
        "texture": "retro_adventure_wall",
        "texrepeat": "2 1",
        "texuniform": "true",
        "reflectance": "0.08",
        "specular": "0.22",
        "shininess": "0.3",
    },
}

OFFSCREEN_WIDTH = 1280
OFFSCREEN_HEIGHT = 820
FLOOR_SENSOR_SEMANTICS = frozenset({"goal", "hazard", "target_region", "floor_switch"})
FLOOR_SENSOR_VISUAL_THICKNESS = 0.10


def compile_spec_to_mjcf(spec: EnvSpec3D | dict[str, Any]) -> str:
    parsed = spec if isinstance(spec, EnvSpec3D) else parse_env_spec_3d(spec)

    root = ET.Element("mujoco", {"model": parsed.id})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.01",
            "gravity": _vec(parsed.gravity),
            "integrator": "RK4",
        },
    )
    _add_visual(root)
    _add_assets(root)
    worldbody = ET.SubElement(root, "worldbody")
    _add_lights(worldbody)
    for camera in parsed.cameras:
        _add_camera(worldbody, camera.model_dump(mode="json"))
    for obj in parsed.objects:
        _add_object(worldbody, obj.model_dump(mode="json"))
    if parsed.mechanisms:
        actuator = ET.SubElement(root, "actuator")
        objects = {obj.id: obj for obj in parsed.objects}
        for mechanism in parsed.mechanisms:
            gate = objects[mechanism.gate_id]
            travel = float(gate.metadata.get("travel") or float(gate.size[2]) + 0.2)
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": f"{gate.id}_actuator",
                    "joint": f"{gate.id}_slide",
                    "kp": "120",
                    "kv": "18",
                    "ctrlrange": f"0 {_fmt(travel)}",
                },
            )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def write_mjcf(spec: EnvSpec3D | dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compile_spec_to_mjcf(spec), encoding="utf-8")
    return path


def validate_mjcf_loads(xml_path: Path) -> dict[str, Any]:
    try:
        import mujoco
    except Exception as exc:  # pragma: no cover - depends on local install
        return {
            "valid": False,
            "error": f"MuJoCo is unavailable: {exc}",
            "dependency_missing": True,
        }
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:
        return {"valid": False, "error": str(exc), "dependency_missing": False}
    return {
        "valid": True,
        "error": "",
        "dependency_missing": False,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "ncam": int(model.ncam),
    }


def _add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "offwidth": str(OFFSCREEN_WIDTH),
            "offheight": str(OFFSCREEN_HEIGHT),
        },
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "4"})
    ET.SubElement(visual, "headlight", {"diffuse": "0.45 0.48 0.38", "ambient": "0.18 0.16 0.12", "specular": "0.24 0.22 0.18"})
    ET.SubElement(visual, "rgba", {"haze": "0.48 0.67 0.82 1"})


def _add_assets(root: ET.Element) -> None:
    asset = ET.SubElement(root, "asset")
    for name, attrs in TEXTURES.items():
        ET.SubElement(asset, "texture", {"name": name, **attrs})
    for name, attrs in MATERIALS.items():
        material_attrs = {"name": name, **attrs}
        material_attrs["rgba"] = _rgba(material_attrs["rgba"])
        ET.SubElement(asset, "material", material_attrs)


def _add_lights(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": "-4 -7 10",
            "dir": "0.35 0.45 -1",
            "diffuse": "1 0.9 0.68",
            "ambient": "0.24 0.2 0.15",
            "specular": "0.32 0.28 0.18",
            "directional": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill_light",
            "pos": "-5 5 5",
            "diffuse": "0.38 0.55 0.75",
            "ambient": "0.1 0.12 0.14",
            "specular": "0.08 0.1 0.12",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "rim_light",
            "pos": "5 -4 6",
            "diffuse": "0.7 0.58 0.35",
            "ambient": "0.06 0.05 0.03",
            "specular": "0.16 0.14 0.08",
        },
    )


def _add_camera(worldbody: ET.Element, camera: dict[str, Any]) -> None:
    x_axis, y_axis = _camera_xyaxes(camera["position"], camera["target"])
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": camera["id"],
            "pos": _vec(camera["position"]),
            "xyaxes": f"{_vec(x_axis)} {_vec(y_axis)}",
            "fovy": _fmt(camera["fov"]),
        },
    )


def _add_object(worldbody: ET.Element, obj: dict[str, Any]) -> None:
    attrs = _geom_attrs(obj)
    if obj["body_type"] == "mechanism":
        travel = float(obj.get("metadata", {}).get("travel") or float(obj["size"][2]) + 0.2)
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obj["id"],
                "pos": _vec(obj["position"]),
                "euler": _body_euler(obj),
                "gravcomp": "1",
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{obj['id']}_slide",
                "type": "slide",
                "axis": "0 0 1",
                "range": f"0 {_fmt(travel)}",
                "damping": "12",
            },
        )
        attrs["pos"] = "0 0 0"
        attrs.pop("euler", None)
        attrs["mass"] = "8"
        ET.SubElement(body, "geom", attrs)
    elif obj["body_type"] == "dynamic":
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obj["id"],
                "pos": _vec(obj["position"]),
                "euler": _body_euler(obj),
            },
        )
        ET.SubElement(body, "freejoint", {"name": f"{obj['id']}_freejoint"})
        attrs["pos"] = "0 0 0"
        attrs.pop("euler", None)
        ET.SubElement(body, "geom", attrs)
    else:
        ET.SubElement(worldbody, "geom", attrs)


def _geom_attrs(obj: dict[str, Any]) -> dict[str, str]:
    shape = obj["shape"]
    semantic_type = obj["semantic_type"]
    attrs: dict[str, str] = {
        "name": obj["id"],
        "material": _material_for(obj),
    }
    if shape == "box":
        position = obj["position"]
        size = obj["size"]
        if visual_box := floor_sensor_visual_box(obj):
            position, size = visual_box
        attrs.update(
            {
                "type": "box",
                "pos": _vec(position),
                "size": _vec([size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]),
                "euler": _body_euler(obj),
            }
        )
    elif shape == "ramp":
        ramp = ramp_geometry_from_object(obj)
        attrs.update(
            {
                "type": "box",
                "pos": _vec(ramp.center),
                "size": _vec([ramp.slope_length / 2.0, ramp.width / 2.0, ramp.thickness / 2.0]),
                "xyaxes": _vec([*ramp.x_axis, *ramp.y_axis]),
            }
        )
    elif shape == "sphere":
        attrs.update({"type": "sphere", "pos": _vec(obj["position"]), "size": _fmt(obj["size"][0] / 2.0)})
    elif shape == "cylinder":
        attrs.update(
            {
                "type": "cylinder",
                "pos": _vec(obj["position"]),
                "size": _vec([obj["size"][0] / 2.0, obj["size"][2] / 2.0]),
            }
        )
    elif shape == "capsule":
        attrs.update(
            {
                "type": "capsule",
                "pos": _vec(obj["position"]),
                "size": _vec([obj["size"][0] / 2.0, obj["size"][2] / 2.0]),
            }
        )
    else:
        raise MuJoCoCompileError(f"unsupported shape {shape!r}")

    if obj["body_type"] == "sensor":
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
    if obj["body_type"] == "dynamic":
        attrs["mass"] = _fmt(1.5 if semantic_type == "agent" else 1.0)
    if semantic_type == "agent":
        attrs["group"] = "5"
    if not obj.get("visible", True):
        attrs["rgba"] = "0 0 0 0"
    return attrs


def floor_sensor_visual_box(
    obj: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    """Return thin visual geometry for a floor sensor's authored footprint.

    The full authored volume remains authoritative for semantic overlap checks.
    MuJoCo sensor geoms are non-colliding, so this only changes their rendering.
    """

    if (
        obj.get("shape") != "box"
        or obj.get("body_type") != "sensor"
        or obj.get("semantic_type") not in FLOOR_SENSOR_SEMANTICS
    ):
        return None
    position = [float(value) for value in obj["position"]]
    size = [float(value) for value in obj["size"]]
    thickness = min(size[2], FLOOR_SENSOR_VISUAL_THICKNESS)
    lower_z = position[2] - size[2] / 2.0
    return (
        [position[0], position[1], lower_z + thickness / 2.0],
        [size[0], size[1], thickness],
    )


def _material_for(obj: dict[str, Any]) -> str:
    semantic = obj["semantic_type"]
    if semantic in MATERIALS:
        return semantic
    return {
        "ground": "ground",
        "wall": "wall",
        "platform": "platform",
        "ramp": "ramp",
        "static_box": "pushable_box",
        "pushable_box": "pushable_box",
        "ball": "ball",
        "cylinder": "cylinder",
        "agent": "agent",
        "goal": "goal",
        "target_region": "target_region",
        "hazard": "hazard",
        "floor_switch": "floor_switch",
        "gate": "gate",
    }.get(semantic, "platform")


def _body_euler(obj: dict[str, Any]) -> str:
    return _vec([0.0, 0.0, float(obj.get("yaw") or 0.0)])


def _camera_xyaxes(position: list[float], target: list[float]) -> tuple[list[float], list[float]]:
    pos = [float(value) for value in position]
    tgt = [float(value) for value in target]
    backward = _normalize([pos[0] - tgt[0], pos[1] - tgt[1], pos[2] - tgt[2]])
    up = [0.0, 0.0, 1.0]
    x_axis = _normalize(_cross(up, backward))
    if _length(x_axis) < 1e-8:
        x_axis = [1.0, 0.0, 0.0]
    y_axis = _normalize(_cross(backward, x_axis))
    return x_axis, y_axis


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(value: list[float]) -> list[float]:
    length = _length(value)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [item / length for item in value]


def _length(value: list[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _rgba(color: str) -> str:
    raw = color.lstrip("#")
    if len(raw) == 6:
        raw += "FF"
    values = [int(raw[index : index + 2], 16) / 255.0 for index in range(0, 8, 2)]
    return _vec(values)


def _vec(values: list[float]) -> str:
    return " ".join(_fmt(float(value)) for value in values)


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"
