"""Agent-facing object catalog for Environment Generation."""

from __future__ import annotations

from typing import Any


OBJECT_CATALOG_3D: dict[str, Any] = {
    "catalog_version": "1.1",
    "principle": (
        "Build robot courtyard scenes from a fixed set of MuJoCo-backed semantic primitives. "
        "New Studio scenes start with only a ground and four walls; agent, goal, mechanisms, "
        "and interior obstacles are added only when requested."
    ),
    "coordinates": {
        "x": "right",
        "y": "forward",
        "z": "up",
    },
    "theme": "robot_courtyard: one sunny low-poly courtyard universe with local assets and fixed physics semantics",
    "primitives": [
        {"name": "ground_plane", "tool": "add_ground_plane", "description": "Bounded floor slab."},
        {"name": "wall", "tool": "add_wall", "description": "Static vertical box barrier."},
        {"name": "platform", "tool": "add_platform", "description": "Static raised box surface."},
        {
            "name": "ramp",
            "tool": "add_ramp",
            "edit_tool": "set_ramp_geometry",
            "description": (
                "Static incline authored from its low walkable endpoint; yaw points uphill, "
                "length is horizontal run, rise is vertical gain, and thickness is collider thickness."
            ),
        },
        {
            "name": "static_box",
            "tool": "add_static_box",
            "description": "Anchored crate-shaped collider for stable towers and structures.",
        },
        {"name": "pushable_box", "tool": "add_pushable_box", "description": "Dynamic box with a free joint."},
        {"name": "ball", "tool": "add_ball", "description": "Dynamic sphere."},
        {"name": "cylinder", "tool": "add_cylinder", "description": "Dynamic upright cylinder."},
        {"name": "agent_spawn", "tool": "add_agent_spawn", "description": "Visible controllable-agent spawn marker."},
        {"name": "goal_zone", "tool": "add_goal_zone", "description": "Visible non-colliding success region."},
        {"name": "hazard_zone", "tool": "add_hazard_zone", "description": "Visible non-colliding hazard region."},
        {"name": "target_region", "tool": "add_target_region", "description": "Neutral sensor region for delivery and task objectives."},
        {"name": "floor_switch", "tool": "add_floor_switch", "description": "Latch-open floor trigger."},
        {"name": "sliding_gate", "tool": "add_sliding_gate", "description": "Vertically opening mechanism barrier."},
    ],
    "recipes": [
        {
            "name": "courtyard_level",
            "tool": "make_courtyard_level",
            "description": "Seeded playable courtyard with barrier, slalom, push, elevation, switch-gate, or mixed layout.",
        },
        {"name": "empty_room", "tool": "make_empty_room", "description": "Floor plus four low boundary walls."},
        {"name": "ramp_course", "tool": "make_ramp_course", "description": "Floor, ramp, platform, agent, and goal."},
        {"name": "box_goal_scene", "tool": "make_box_goal_scene", "description": "Floor, wall, pushable box, agent, and goal."},
    ],
}
