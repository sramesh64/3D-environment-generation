from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from .schema import EnvSpec3D, parse_env_spec_3d


VISUAL_SCENE_FILENAME = "visual_scene.json"
VISUAL_SCENE_VERSION = "1.0"
DEFAULT_VISUAL_THEME = "storybook_adventure"
COURTYARD_VISUAL_THEME = "robot_courtyard"


def compile_visual_scene(spec: EnvSpec3D | dict[str, Any]) -> dict[str, Any]:
    parsed = spec if isinstance(spec, EnvSpec3D) else parse_env_spec_3d(spec)
    objects = [obj.model_dump(mode="json") for obj in parsed.objects]
    visual_objects = [_visual_object_for(obj) for obj in objects if obj.get("visible", True)]
    theme = parsed.theme if parsed.theme in {DEFAULT_VISUAL_THEME, COURTYARD_VISUAL_THEME} else DEFAULT_VISUAL_THEME
    decoration_seed = "robot_courtyard_perimeter" if theme == COURTYARD_VISUAL_THEME else parsed.id
    decorations = _decorations_for(decoration_seed, parsed.world_size, objects)
    return {
        "schema_version": VISUAL_SCENE_VERSION,
        "source_env_id": parsed.id,
        "theme": theme,
        "world_size": list(parsed.world_size),
        "units": "meters",
        "environment": {
            "sky": {
                "style": "courtyard_daylight" if theme == COURTYARD_VISUAL_THEME else "storybook_sky",
                "sun": True,
            },
            "set_dressing": {
                "placement": "perimeter_only",
                "clear_play_area": True,
                "grounding": "continuous_meadow",
            },
        },
        "camera": _camera_for(parsed.world_size, objects),
        "palette": _palette(theme),
        "objects": visual_objects + decorations,
        "metadata": {
            "source": "env_spec_3d.json",
            "derived": True,
            "decorations_are_visual_only": True,
            "visual_decorations_avoid_play_area": True,
        },
    }


def _visual_object_for(obj: dict[str, Any]) -> dict[str, Any]:
    semantic = obj["semantic_type"]
    visual_type = {
        "ground": "terrain_grass",
        "wall": _wall_visual_type(obj),
        "platform": "wood_platform",
        "ramp": "wood_ramp",
        "static_box": "crate",
        "pushable_box": "crate",
        "ball": "boulder",
        "cylinder": "barrel",
        "agent": "agent_hero",
        "goal": "goal_portal",
        "target_region": "target_region",
        "hazard": "hazard_spikes",
        "floor_switch": "floor_switch",
        "gate": "sliding_gate",
    }.get(semantic, "storybook_prop")
    appearance = dict(obj.get("appearance") or {})
    if appearance.get("asset_id") == "courtyard_goal_pad":
        visual_type = "goal_pad"
    elif appearance.get("asset_id") == "courtyard_target_region":
        visual_type = "target_region"
    elif appearance.get("asset_id") == "courtyard_hazard":
        visual_type = "courtyard_hazard"
    return {
        "id": f"visual_{obj['id']}",
        "source_id": obj["id"],
        "physics_backed": True,
        "semantic_type": semantic,
        "visual_type": visual_type,
        "position": list(obj["position"]),
        "size": list(obj["size"]),
        "yaw": float(obj.get("yaw") or 0.0),
        "body_type": obj["body_type"],
        "tags": list(obj.get("tags") or []),
        "appearance": appearance or None,
        "metadata": dict(obj.get("metadata") or {}),
    }


def _decorations_for(env_id: str, world_size: list[float], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ground = _primary_ground(objects)
    if ground is None:
        return []
    width = float(ground["size"][0]) if ground else float(world_size[0])
    depth = float(ground["size"][1]) if ground else float(world_size[1])
    center_x = float(ground["position"][0]) if ground else 0.0
    center_y = float(ground["position"][1]) if ground else 0.0
    rng = _rng(env_id, "storybook_decor")
    decorations: list[dict[str, Any]] = []

    tree_slots = [
        ("south", -0.44, 1.45),
        ("south", 0.0, 1.85),
        ("south", 0.42, 1.35),
        ("north", -0.34, 1.7),
        ("north", 0.36, 1.5),
        ("west", -0.24, 1.55),
        ("west", 0.34, 1.8),
        ("east", 0.12, 1.55),
    ]
    for index, (side, along, offset) in enumerate(tree_slots):
        decorations.append(
            _decor(
                env_id,
                "tree",
                index,
                _perimeter_position(center_x, center_y, width, depth, side, along, offset, rng, jitter=0.08),
                [0.9 + rng.random() * 0.45, 0.9 + rng.random() * 0.45, 2.4 + rng.random() * 0.8],
                yaw=rng.uniform(-math.pi, math.pi),
                variant=rng.choice(["round", "pine", "two_tier"]),
            )
        )

    shrub_slots = [
        ("south", -0.28, 0.8),
        ("south", 0.28, 0.9),
        ("north", -0.08, 0.8),
        ("north", 0.44, 0.95),
        ("west", 0.04, 0.85),
        ("east", -0.36, 0.9),
    ]
    for index, (side, along, offset) in enumerate(shrub_slots):
        decorations.append(
            _decor(
                env_id,
                "shrub",
                index,
                _perimeter_position(center_x, center_y, width, depth, side, along, offset, rng, jitter=0.08),
                [0.65 + rng.random() * 0.25, 0.55 + rng.random() * 0.25, 0.45 + rng.random() * 0.2],
                yaw=rng.uniform(-math.pi, math.pi),
            )
        )

    for index in range(5):
        side = rng.choice(["north", "south", "east", "west"])
        decorations.append(
            _decor(
                env_id,
                "stone",
                index,
                _perimeter_position(center_x, center_y, width, depth, side, rng.uniform(-0.42, 0.42), 0.55, rng, jitter=0.04),
                [0.35 + rng.random() * 0.35, 0.3 + rng.random() * 0.25, 0.22 + rng.random() * 0.18],
                yaw=rng.uniform(-math.pi, math.pi),
            )
        )

    return decorations


def _perimeter_position(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    side: str,
    along: float,
    offset: float,
    rng: random.Random,
    *,
    jitter: float,
) -> list[float]:
    t = max(-0.48, min(0.48, along + rng.uniform(-jitter, jitter)))
    if side == "north":
        return [center_x + t * width, center_y + depth * 0.5 + offset, 0.0]
    if side == "south":
        return [center_x + t * width, center_y - depth * 0.5 - offset, 0.0]
    if side == "east":
        return [center_x + width * 0.5 + offset, center_y + t * depth, 0.0]
    return [center_x - width * 0.5 - offset, center_y + t * depth, 0.0]


def _decor(
    env_id: str,
    visual_type: str,
    index: int,
    position: list[float],
    size: list[float],
    *,
    yaw: float = 0.0,
    variant: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    metadata = {"seed": _stable_seed(env_id, visual_type, str(index))}
    if variant:
        metadata["variant"] = variant
    appearances = {
        "tree": {"asset_id": "courtyard_tree", "variant": variant or "round"},
        "shrub": {"asset_id": "courtyard_shrub"},
        "stone": {"asset_id": "courtyard_rock"},
        "sign": {"asset_id": "courtyard_sign"},
    }
    return {
        "id": f"decor_{visual_type}_{index:02d}",
        "source_id": source_id,
        "physics_backed": False,
        "semantic_type": "decoration",
        "visual_type": visual_type,
        "position": position,
        "size": size,
        "yaw": yaw,
        "body_type": "visual",
        "tags": ["visual_only", visual_type],
        "appearance": appearances.get(visual_type),
        "metadata": metadata,
    }


def _camera_for(world_size: list[float], objects: list[dict[str, Any]]) -> dict[str, Any]:
    ground = _primary_ground(objects)
    if ground:
        target = [float(ground["position"][0]), float(ground["position"][1]), 0.5]
        extent = max(float(ground["size"][0]), float(ground["size"][1]))
    else:
        target = [0.0, 0.0, 0.5]
        extent = max(float(world_size[0]), float(world_size[1]))
    return {
        "target": target,
        "distance": max(9.0, extent * 0.95),
        "azimuth": -42.0,
        "elevation": 38.0,
        "min_distance": max(4.5, extent * 0.35),
        "max_distance": max(16.0, extent * 1.65),
    }


def _palette(theme: str = DEFAULT_VISUAL_THEME) -> dict[str, str]:
    palette = {
        "sky_top": "#8ED6F7",
        "sky_bottom": "#DDF2FF",
        "grass": "#69B85D",
        "grass_light": "#8FD277",
        "grass_dark": "#397B48",
        "meadow": "#63A957",
        "meadow_dark": "#4E8947",
        "clearing": "#7BCB67",
        "clearing_alt": "#A8DD82",
        "path": "#C9B27C",
        "wood": "#A96D3B",
        "wood_dark": "#5D3525",
        "stone": "#A9A8A2",
        "stone_dark": "#686D70",
        "leaf": "#4AA35A",
        "leaf_light": "#7ED16A",
        "agent": "#3A7BFF",
        "goal": "#FFD166",
        "goal_core": "#66E3FF",
        "target": "#70D6E8",
        "switch": "#4AC7E8",
        "hazard": "#E95858",
        "shadow": "#172019",
    }
    if theme == COURTYARD_VISUAL_THEME:
        palette.update({"sky_top": "#78C9F1", "sky_bottom": "#E9F6FF", "stone": "#B8B6A9"})
    return palette


def _wall_visual_type(obj: dict[str, Any]) -> str:
    variant = str((obj.get("appearance") or {}).get("variant") or "")
    if variant in {"fence", "hedge", "stone", "plain"}:
        return {
            "fence": "fence_wall",
            "hedge": "hedge_wall",
            "stone": "stone_wall",
            "plain": "plain_wall",
        }[variant]
    if "boundary" in (obj.get("tags") or []):
        return "plain_wall"
    seed = _stable_seed(obj["id"], "wall")
    if seed % 3 == 0:
        return "fence_wall"
    if seed % 3 == 1:
        return "hedge_wall"
    return "stone_wall"


def _primary_ground(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    grounds = [obj for obj in objects if obj["semantic_type"] == "ground"]
    if not grounds:
        return None
    return max(grounds, key=lambda item: float(item["size"][0]) * float(item["size"][1]))


def _rng(*parts: str) -> random.Random:
    return random.Random(_stable_seed(*parts))


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
