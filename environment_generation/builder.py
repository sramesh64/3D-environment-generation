from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any, Sequence

from .ramp_geometry import make_ramp_geometry, ramp_geometry_from_object, ramp_object_fields
from .schema import EnvObject3D, EnvSpec3D, validate_env_spec_3d


Number = int | float
Vec3Input = Sequence[Number]


class BuilderOperationError(ValueError):
    """Raised when a semantic 3D edit is malformed."""


class BuilderValidationError(ValueError):
    """Raised when a draft cannot be finalized."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("Draft environment is not finalizable:\n- " + "\n- ".join(issues))


DEFAULT_PALETTE: dict[str, str] = {
    "ground": "#8BD66C",
    "wall": "#28746F",
    "platform": "#C6974C",
    "ramp": "#C6974C",
    "static_box": "#D68A3A",
    "pushable_box": "#EF6430",
    "ball": "#F0C84A",
    "cylinder": "#DF7030",
    "agent": "#2D73FF",
    "goal": "#57F287",
    "target_region": "#70D6E8",
    "hazard": "#D9365E",
    "floor_switch": "#49C6E5",
    "gate": "#73806F",
}


class EnvSpec3DBuilder:
    """Build an EnvSpec3D through small semantic operations."""

    def __init__(
        self,
        env_id: str = "studio_env_001",
        *,
        description: str = "A generated 3D MuJoCo environment.",
    ) -> None:
        self._validate_id(env_id, "env_id")
        self.env_id = env_id
        self.description = self._clean_text(description, "description")
        self._id_counters: Counter[str] = Counter()
        self._claimed_ids: set[str] = set()
        self._objects: list[dict[str, Any]] = []
        self.schema_version = "1.0"
        self.world_size = [24.0, 18.0, 8.0]
        self.gravity = [0.0, 0.0, -9.81]
        self.theme = "storybook_adventure"
        self.game: dict[str, Any] | None = None
        self.generation: dict[str, Any] | None = None
        self.mechanisms: list[dict[str, Any]] = []
        self._cameras = self._default_cameras()

    @classmethod
    def from_spec_dict(cls, value: dict[str, Any]) -> "EnvSpec3DBuilder":
        spec = EnvSpec3D.model_validate(value)
        builder = cls(spec.id, description=spec.description)
        builder.schema_version = spec.schema_version
        builder.world_size = list(spec.world_size)
        builder.gravity = list(spec.gravity)
        builder.theme = spec.theme
        builder._cameras = [camera.model_dump(mode="json") for camera in spec.cameras]
        builder._objects = [obj.model_dump(mode="json", exclude_none=True) for obj in spec.objects]
        builder.game = spec.game.model_dump(mode="json") if spec.game else None
        builder.generation = spec.generation.model_dump(mode="json") if spec.generation else None
        builder.mechanisms = [item.model_dump(mode="json") for item in spec.mechanisms]
        builder._claimed_ids = {obj["id"] for obj in builder._objects}
        return builder

    def set_world(
        self,
        width: Number = 24,
        depth: Number = 18,
        height: Number = 8,
        gravity: Vec3Input | None = None,
        theme: str | None = None,
    ) -> dict[str, Any]:
        if self._objects:
            raise BuilderOperationError("set_world must be called before adding scene objects")
        self.world_size = [
            self._positive(width, "width"),
            self._positive(depth, "depth"),
            self._positive(height, "height"),
        ]
        self.gravity = self._vec3(gravity if gravity is not None else [0, 0, -9.81], "gravity")
        if theme and theme.strip():
            cleaned_theme = theme.strip().lower()
            if cleaned_theme not in {"storybook_adventure", "robot_courtyard"}:
                raise BuilderOperationError("theme must be robot_courtyard or storybook_adventure")
            self.theme = cleaned_theme
        self._cameras = self._default_cameras()
        return self.inspect_draft()

    def set_description(self, description: str) -> dict[str, Any]:
        self.description = self._clean_text(description, "description")
        return self.inspect_draft()

    def move_object(
        self,
        id: str,
        x: Number | None = None,
        y: Number | None = None,
        z: Number | None = None,
    ) -> dict[str, Any]:
        if x is None and y is None and z is None:
            raise BuilderOperationError("move_object requires at least one coordinate")
        obj = self._object_by_id(id)
        if obj["shape"] == "ramp":
            raise BuilderOperationError(
                "ramps must be moved with set_ramp_geometry; x, y, and z are the low walkable endpoint"
            )
        position = list(obj["position"])
        if x is not None:
            position[0] = self._number(x, "x")
        if y is not None:
            position[1] = self._number(y, "y")
        if z is not None:
            position[2] = self._number(z, "z")
        obj["position"] = position
        self._replace_object(obj)
        self._update_linked_camera_for_object(obj)
        return self.inspect_draft()

    def rotate_object(self, id: str, yaw: Number) -> dict[str, Any]:
        obj = self._object_by_id(id)
        if obj["shape"] == "ramp":
            raise BuilderOperationError(
                "ramps must be rotated with set_ramp_geometry; yaw points uphill from the low endpoint"
            )
        obj["yaw"] = self._number(yaw, "yaw")
        self._replace_object(obj)
        return self.inspect_draft()

    def set_object_appearance(
        self,
        id: str,
        asset_id: str,
        variant: str | None = None,
    ) -> dict[str, Any]:
        obj = self._object_by_id(id)
        appearance: dict[str, Any] = {"asset_id": self._clean_id(asset_id, "asset_id")}
        if variant is not None:
            appearance["variant"] = self._clean_id(variant, "variant")
        obj["appearance"] = appearance
        self._replace_object(obj)
        return self.inspect_draft()

    def resize_object(
        self,
        id: str,
        width: Number | None = None,
        depth: Number | None = None,
        height: Number | None = None,
    ) -> dict[str, Any]:
        if width is None and depth is None and height is None:
            raise BuilderOperationError("resize_object requires at least one dimension")
        obj = self._object_by_id(id)
        if obj["shape"] == "ramp":
            raise BuilderOperationError(
                "ramps must be resized with set_ramp_geometry; use length, width, rise, and thickness"
            )
        old_size = list(obj["size"])
        old_bottom_z = float(obj["position"][2]) - old_size[2] / 2.0
        new_size = [
            self._positive(width, "width") if width is not None else old_size[0],
            self._positive(depth, "depth") if depth is not None else old_size[1],
            self._positive(height, "height") if height is not None else old_size[2],
        ]
        if obj["shape"] == "sphere":
            provided = [value for value in (width, depth, height) if value is not None]
            diameter = self._positive(provided[0], "diameter") if provided else old_size[0]
            if any(abs(self._positive(value, "diameter") - diameter) > 1e-9 for value in provided):
                raise BuilderOperationError("sphere resize dimensions must match")
            new_size = [diameter, diameter, diameter]
        elif obj["shape"] in {"cylinder", "capsule"}:
            if width is not None and depth is None:
                new_size[1] = new_size[0]
            elif depth is not None and width is None:
                new_size[0] = new_size[1]
            if abs(new_size[0] - new_size[1]) > 1e-9:
                raise BuilderOperationError(f"{obj['shape']} resize width and depth must match")
        obj["size"] = new_size
        obj["position"] = [obj["position"][0], obj["position"][1], old_bottom_z + new_size[2] / 2.0]
        self._replace_object(obj)
        self._update_linked_camera_for_object(obj)
        return self.inspect_draft()

    def remove_object(self, id: str) -> dict[str, Any]:
        obj = self._object_by_id(id)
        self._objects = [item for item in self._objects if item["id"] != obj["id"]]
        self._claimed_ids.discard(obj["id"])
        if self.game and obj["id"] in {self.game.get("agent_id"), self.game.get("goal_id")}:
            self.game = None
        return self.inspect_draft()

    def add_ground_plane(
        self,
        width: Number | None = None,
        depth: Number | None = None,
        thickness: Number = 0.2,
        id: str | None = None,
    ) -> str:
        width_f = self._positive(width if width is not None else self.world_size[0], "width")
        depth_f = self._positive(depth if depth is not None else self.world_size[1], "depth")
        thickness_f = self._positive(thickness, "thickness")
        object_id = self._allocate_id("ground", requested=id)
        self._add_object(
            id=object_id,
            semantic_type="ground",
            shape="box",
            body_type="static",
            position=[0.0, 0.0, -thickness_f / 2.0],
            size=[width_f, depth_f, thickness_f],
            color=DEFAULT_PALETTE["ground"],
            label="Ground",
            tags=["floor", "walkable"],
        )
        return object_id

    def add_wall(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 0.4,
        depth: Number = 6,
        height: Number = 2.5,
        yaw: Number = 0,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("wall", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="wall",
            shape="box",
            body_type="static",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            yaw=self._number(yaw, "yaw"),
            color=DEFAULT_PALETTE["wall"],
            label="Wall",
            tags=["barrier"],
        )
        return object_id

    def add_platform(
        self,
        x: Number,
        y: Number,
        z: Number,
        width: Number = 4,
        depth: Number = 4,
        thickness: Number = 0.3,
        yaw: Number = 0,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("platform", requested=id)
        thickness_f = self._positive(thickness, "thickness")
        self._add_object(
            id=object_id,
            semantic_type="platform",
            shape="box",
            body_type="static",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + thickness_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), thickness_f],
            yaw=self._number(yaw, "yaw"),
            color=DEFAULT_PALETTE["platform"],
            label="Platform",
            tags=["walkable"],
        )
        return object_id

    def add_ramp(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        length: Number = 4,
        width: Number = 2,
        rise: Number = 1.5,
        thickness: Number = 0.25,
        yaw: Number = 0,
        id: str | None = None,
        *,
        height: Number | None = None,
    ) -> str:
        if height is not None:
            if float(rise) != 1.5:
                raise BuilderOperationError("add_ramp accepts rise, not both rise and the legacy height alias")
            rise = height
        geometry = make_ramp_geometry(
            low_end=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z")],
            length=self._positive(length, "length"),
            width=self._positive(width, "width"),
            rise=self._positive(rise, "rise"),
            thickness=self._positive(thickness, "thickness"),
            yaw=self._number(yaw, "yaw"),
        )
        object_id = self._allocate_id("ramp", requested=id)
        fields = ramp_object_fields(geometry)
        self._add_object(
            id=object_id,
            semantic_type="ramp",
            shape="ramp",
            body_type="static",
            position=fields["position"],
            size=fields["size"],
            yaw=fields["yaw"],
            color=DEFAULT_PALETTE["ramp"],
            label="Ramp",
            tags=["walkable", "incline"],
            metadata=fields["metadata"],
        )
        return object_id

    def set_ramp_geometry(
        self,
        id: str,
        x: Number | None = None,
        y: Number | None = None,
        z: Number | None = None,
        length: Number | None = None,
        width: Number | None = None,
        rise: Number | None = None,
        thickness: Number | None = None,
        yaw: Number | None = None,
    ) -> dict[str, Any]:
        if all(value is None for value in (x, y, z, length, width, rise, thickness, yaw)):
            raise BuilderOperationError("set_ramp_geometry requires at least one geometry field")
        obj = self._object_by_id(id)
        if obj["shape"] != "ramp":
            raise BuilderOperationError("set_ramp_geometry requires an existing ramp object")
        current = ramp_geometry_from_object(obj)
        low_end = [
            self._number(x, "x") if x is not None else current.low_end[0],
            self._number(y, "y") if y is not None else current.low_end[1],
            self._number(z, "z") if z is not None else current.low_end[2],
        ]
        geometry = make_ramp_geometry(
            low_end=low_end,
            length=self._positive(length, "length") if length is not None else current.length,
            width=self._positive(width, "width") if width is not None else current.width,
            rise=self._positive(rise, "rise") if rise is not None else current.rise,
            thickness=(
                self._positive(thickness, "thickness") if thickness is not None else current.thickness
            ),
            yaw=self._number(yaw, "yaw") if yaw is not None else current.yaw,
        )
        obj.update(ramp_object_fields(geometry, metadata=obj.get("metadata")))
        self._replace_object(obj)
        return self.inspect_draft()

    def add_pushable_box(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 1,
        depth: Number = 1,
        height: Number = 1,
        yaw: Number = 0,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("box", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="pushable_box",
            shape="box",
            body_type="dynamic",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            yaw=self._number(yaw, "yaw"),
            color=DEFAULT_PALETTE["pushable_box"],
            label="Pushable box",
            tags=["pushable"],
        )
        return object_id

    def add_static_box(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 1,
        depth: Number = 1,
        height: Number = 1,
        yaw: Number = 0,
        id: str | None = None,
    ) -> str:
        """Add an anchored crate-shaped collider for stable structures."""

        object_id = self._allocate_id("static_box", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="static_box",
            shape="box",
            body_type="static",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            yaw=self._number(yaw, "yaw"),
            color=DEFAULT_PALETTE["static_box"],
            label="Static box",
            tags=["box", "crate", "anchored", "structure"],
        )
        return object_id

    def add_ball(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        radius: Number = 0.5,
        id: str | None = None,
    ) -> str:
        radius_f = self._positive(radius, "radius")
        object_id = self._allocate_id("ball", requested=id)
        diameter = radius_f * 2.0
        self._add_object(
            id=object_id,
            semantic_type="ball",
            shape="sphere",
            body_type="dynamic",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + radius_f],
            size=[diameter, diameter, diameter],
            color=DEFAULT_PALETTE["ball"],
            label="Ball",
            tags=["pushable", "rolling"],
        )
        return object_id

    def add_cylinder(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        radius: Number = 0.45,
        height: Number = 1,
        id: str | None = None,
    ) -> str:
        radius_f = self._positive(radius, "radius")
        height_f = self._positive(height, "height")
        object_id = self._allocate_id("cylinder", requested=id)
        diameter = radius_f * 2.0
        self._add_object(
            id=object_id,
            semantic_type="cylinder",
            shape="cylinder",
            body_type="dynamic",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[diameter, diameter, height_f],
            color=DEFAULT_PALETTE["cylinder"],
            label="Cylinder",
            tags=["pushable", "rolling"],
        )
        return object_id

    def add_agent_spawn(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        radius: Number = 0.35,
        height: Number = 1.1,
        id: str | None = None,
    ) -> str:
        if any(obj["semantic_type"] == "agent" for obj in self._objects):
            raise BuilderOperationError("only one agent_spawn is supported in v1")
        radius_f = self._positive(radius, "radius")
        height_f = self._positive(height, "height")
        object_id = self._allocate_id("agent", requested=id)
        diameter = radius_f * 2.0
        self._add_object(
            id=object_id,
            semantic_type="agent",
            shape="cylinder",
            body_type="dynamic",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[diameter, diameter, height_f],
            color=DEFAULT_PALETTE["agent"],
            label="Agent spawn",
            tags=["agent", "spawn"],
        )
        self._update_agent_camera(self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0)
        return object_id

    def add_goal_zone(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 1.4,
        depth: Number = 1.4,
        height: Number = 1.2,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("goal", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="goal",
            shape="box",
            body_type="sensor",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            color=DEFAULT_PALETTE["goal"],
            label="Goal zone",
            tags=["goal", "sensor"],
        )
        self._update_goal_camera(self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0)
        return object_id

    def add_hazard_zone(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 2,
        depth: Number = 2,
        height: Number = 0.15,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("hazard", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="hazard",
            shape="box",
            body_type="sensor",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            color=DEFAULT_PALETTE["hazard"],
            label="Hazard zone",
            tags=["hazard", "sensor"],
        )
        return object_id

    def add_target_region(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 1.8,
        depth: Number = 1.8,
        height: Number = 0.2,
        id: str | None = None,
    ) -> str:
        """Add a neutral sensor region for task-defined object or agent delivery."""

        object_id = self._allocate_id("target_region", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="target_region",
            shape="box",
            body_type="sensor",
            position=[
                self._number(x, "x"),
                self._number(y, "y"),
                self._number(z, "z") + height_f / 2.0,
            ],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            color=DEFAULT_PALETTE["target_region"],
            label="Target region",
            tags=["target", "delivery", "sensor"],
            appearance={"asset_id": "courtyard_target_region"},
        )
        return object_id

    def add_floor_switch(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 1.2,
        depth: Number = 1.2,
        height: Number = 0.12,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("floor_switch", requested=id)
        height_f = self._positive(height, "height")
        self._add_object(
            id=object_id,
            semantic_type="floor_switch",
            shape="box",
            body_type="sensor",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            color=DEFAULT_PALETTE["floor_switch"],
            label="Floor switch",
            tags=["trigger", "sensor", "mechanism"],
            appearance={"asset_id": "courtyard_floor_switch"},
        )
        return object_id

    def add_sliding_gate(
        self,
        x: Number,
        y: Number,
        z: Number = 0,
        width: Number = 3.0,
        depth: Number = 0.3,
        height: Number = 1.8,
        yaw: Number = 0,
        travel: Number | None = None,
        id: str | None = None,
    ) -> str:
        object_id = self._allocate_id("gate", requested=id)
        height_f = self._positive(height, "height")
        travel_f = self._positive(travel if travel is not None else height_f + 0.2, "travel")
        self.schema_version = "1.1"
        self._add_object(
            id=object_id,
            semantic_type="gate",
            shape="box",
            body_type="mechanism",
            position=[self._number(x, "x"), self._number(y, "y"), self._number(z, "z") + height_f / 2.0],
            size=[self._positive(width, "width"), self._positive(depth, "depth"), height_f],
            yaw=self._number(yaw, "yaw"),
            color=DEFAULT_PALETTE["gate"],
            label="Sliding gate",
            tags=["barrier", "gate", "mechanism"],
            appearance={"asset_id": "courtyard_gate"},
            metadata={"travel": travel_f},
        )
        return object_id

    def link_switch_to_gate(
        self,
        trigger_id: str,
        gate_id: str,
        id: str | None = None,
    ) -> dict[str, Any]:
        trigger = self._object_by_id(trigger_id)
        gate = self._object_by_id(gate_id)
        if trigger["semantic_type"] != "floor_switch":
            raise BuilderOperationError("trigger_id must reference a floor_switch")
        if gate["semantic_type"] != "gate":
            raise BuilderOperationError("gate_id must reference a gate")
        mechanism_id = self._clean_id(id or f"{trigger_id}_opens_{gate_id}", "mechanism id")
        if any(item["id"] == mechanism_id for item in self.mechanisms):
            raise BuilderOperationError(f"mechanism id {mechanism_id!r} is already in use")
        if any(item["gate_id"] == gate_id for item in self.mechanisms):
            raise BuilderOperationError(f"gate {gate_id!r} is already linked")
        self.schema_version = "1.1"
        self.mechanisms.append(
            {"id": mechanism_id, "trigger_id": trigger_id, "gate_id": gate_id, "mode": "latch_open"}
        )
        return self.inspect_draft()

    def configure_reach_goal_game(
        self,
        agent_id: str,
        goal_id: str,
    ) -> dict[str, Any]:
        agent = self._object_by_id(agent_id)
        goal = self._object_by_id(goal_id)
        if agent["semantic_type"] != "agent":
            raise BuilderOperationError("agent_id must reference an agent object")
        if goal["semantic_type"] != "goal":
            raise BuilderOperationError("goal_id must reference a goal object")
        if sum(obj["semantic_type"] == "agent" for obj in self._objects) != 1:
            raise BuilderOperationError("reach-goal gameplay requires exactly one agent")
        if sum(obj["semantic_type"] == "goal" for obj in self._objects) != 1:
            raise BuilderOperationError("reach-goal gameplay requires exactly one goal")
        grounds = [obj for obj in self._objects if obj["semantic_type"] == "ground"]
        if grounds:
            ground = max(grounds, key=lambda obj: float(obj["size"][0]) * float(obj["size"][1]))
            play_width = float(ground["size"][0])
            play_depth = float(ground["size"][1])
        else:
            play_width = float(self.world_size[0])
            play_depth = float(self.world_size[1])
        self.schema_version = "1.1"
        self.game = {
            "mode": "reach_goal",
            "agent_id": agent["id"],
            "goal_id": goal["id"],
            "play_bounds": [play_width, play_depth, min(float(self.world_size[2]), 8.0)],
            "reset_on": ["out_of_bounds"],
        }
        return self.inspect_draft()

    def make_courtyard_level(
        self,
        family: str = "barrier_route",
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> dict[str, Any]:
        from .courtyard import CourtyardGenerationError, is_courtyard_shell, populate_courtyard_level

        if self._objects and not is_courtyard_shell(self.to_spec_dict()):
            raise BuilderOperationError("courtyard generation requires an empty draft or the standard courtyard shell")

        try:
            populate_courtyard_level(self, family=family, difficulty=difficulty, seed=seed)
        except CourtyardGenerationError as exc:
            raise BuilderOperationError(str(exc)) from exc
        return self.inspect_draft()

    def make_empty_room(
        self,
        width: Number = 16,
        depth: Number = 12,
        wall_height: Number = 1.2,
    ) -> dict[str, Any]:
        if self._objects:
            raise BuilderOperationError("recipes must be the first scene-building operation")
        width_f = self._positive(width, "width")
        depth_f = self._positive(depth, "depth")
        wall_h = self._positive(wall_height, "wall_height")
        self.set_world(width=max(width_f + 4, 18), depth=max(depth_f + 4, 14), height=8)
        self.add_ground_plane(width_f, depth_f, id="ground")
        self.add_wall(0, depth_f / 2, width=width_f, depth=0.25, height=wall_h, id="back_wall")
        self.add_wall(0, -depth_f / 2, width=width_f, depth=0.25, height=wall_h, id="front_wall")
        self.add_wall(-width_f / 2, 0, width=0.25, depth=depth_f, height=wall_h, id="left_wall")
        self.add_wall(width_f / 2, 0, width=0.25, depth=depth_f, height=wall_h, id="right_wall")
        return self.inspect_draft()

    def make_ramp_course(self) -> dict[str, Any]:
        if self._objects:
            raise BuilderOperationError("recipes must be the first scene-building operation")
        self.set_world(width=22, depth=14, height=9)
        self.add_ground_plane(20, 12, id="ground")
        self.add_agent_spawn(-7, -3, id="agent")
        self.add_ramp(-2, -1.5, length=5, width=2.2, rise=1.8, id="main_ramp")
        self.add_platform(2.6, -1.5, z=1.8, width=4, depth=3, id="upper_platform")
        self.add_goal_zone(5.5, -1.5, z=1.8, id="goal")
        return self.inspect_draft()

    def make_box_goal_scene(self) -> dict[str, Any]:
        if self._objects:
            raise BuilderOperationError("recipes must be the first scene-building operation")
        self.set_world(width=22, depth=14, height=8)
        self.add_ground_plane(20, 12, id="ground")
        self.add_agent_spawn(-7, 0, id="agent")
        self.add_wall(0, 0, width=0.35, depth=5.5, height=1.7, id="center_wall")
        self.add_pushable_box(-3.0, 0, width=1.1, depth=1.1, height=1.1, id="pushable_box")
        self.add_goal_zone(6.5, 0, id="goal")
        return self.inspect_draft()

    def inspect_draft(self) -> dict[str, Any]:
        counts: Counter[str] = Counter(obj["semantic_type"] for obj in self._objects)
        return {
            "env_id": self.env_id,
            "description": self.description,
            "world_size": list(self.world_size),
            "object_count": len(self._objects),
            "semantic_counts": dict(sorted(counts.items())),
            "objects": copy.deepcopy(self._objects),
            "cameras": copy.deepcopy(self._cameras),
            "game": copy.deepcopy(self.game),
            "generation": copy.deepcopy(self.generation),
            "mechanisms": copy.deepcopy(self.mechanisms),
            "ready_to_finalize": not self._finalization_issues(),
            "issues": self._finalization_issues(),
        }

    def validate_draft(self) -> dict[str, Any]:
        issues = self._finalization_issues()
        if issues:
            return {"valid": False, "issues": issues}
        try:
            validated = validate_env_spec_3d(self._spec_dict())
            if validated.get("game"):
                from .courtyard import validate_courtyard_layout

                game_issues = validate_courtyard_layout(validated)
                if game_issues:
                    return {"valid": False, "issues": game_issues}
        except Exception as exc:
            return {"valid": False, "issues": [str(exc)]}
        return {"valid": True, "issues": []}

    def finalize(self) -> EnvSpec3D:
        validation = self.validate_draft()
        if not validation["valid"]:
            raise BuilderValidationError(validation["issues"])
        return EnvSpec3D.model_validate(self._spec_dict())

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the current draft as JSON-compatible EnvSpec3D data."""

        return self._spec_dict()

    def _spec_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "id": self.env_id,
            "description": self.description,
            "world_size": list(self.world_size),
            "gravity": list(self.gravity),
            "theme": self.theme,
            "cameras": copy.deepcopy(self._cameras),
            "objects": copy.deepcopy(self._objects),
            "metadata": {
                "generator": "environment_generation_semantic_builder_v1" if self.schema_version == "1.1" else "environment_generation_semantic_builder_v0",
                "coordinate_system": "z_up",
                "dimension_units": "meters",
            },
        }
        if self.schema_version == "1.1":
            value["game"] = copy.deepcopy(self.game)
            value["generation"] = copy.deepcopy(self.generation)
            value["mechanisms"] = copy.deepcopy(self.mechanisms)
        return value

    def _add_object(self, **kwargs: Any) -> None:
        if self.theme == "robot_courtyard" and not kwargs.get("appearance"):
            appearance = self._courtyard_appearance(str(kwargs.get("semantic_type") or ""), str(kwargs.get("id") or ""))
            if appearance:
                kwargs["appearance"] = appearance
        obj = EnvObject3D.model_validate(kwargs).model_dump(mode="json", exclude_none=True)
        self._objects.append(obj)

    @staticmethod
    def _courtyard_appearance(semantic_type: str, _object_id: str) -> dict[str, str] | None:
        if semantic_type == "wall":
            # Adjacent wall segments often form one logical barrier. A stable
            # default keeps those segments visually continuous; callers can
            # still opt into fence or hedge appearances explicitly.
            return {"asset_id": "courtyard_boundary", "variant": "stone"}
        return {
            "ground": {"asset_id": "courtyard_ground", "variant": "grass_pavers"},
            "platform": {"asset_id": "courtyard_platform", "variant": "wood"},
            "ramp": {"asset_id": "courtyard_ramp", "variant": "wood"},
            "static_box": {"asset_id": "courtyard_static_prop", "variant": "crate"},
            "pushable_box": {"asset_id": "courtyard_pushable_crate"},
            "cylinder": {"asset_id": "courtyard_barrel"},
            "agent": {"asset_id": "courtyard_robot"},
            "goal": {"asset_id": "courtyard_goal_pad"},
            "target_region": {"asset_id": "courtyard_target_region"},
            "hazard": {"asset_id": "courtyard_hazard", "variant": "broken_paving"},
            "floor_switch": {"asset_id": "courtyard_floor_switch"},
            "gate": {"asset_id": "courtyard_gate"},
        }.get(semantic_type)

    def _object_by_id(self, object_id: str) -> dict[str, Any]:
        self._validate_id(object_id, "object id")
        for obj in self._objects:
            if obj["id"] == object_id:
                return copy.deepcopy(obj)
        raise BuilderOperationError(f"object {object_id!r} does not exist")

    def _replace_object(self, next_object: dict[str, Any]) -> None:
        validated = EnvObject3D.model_validate(next_object).model_dump(mode="json", exclude_none=True)
        for index, obj in enumerate(self._objects):
            if obj["id"] == validated["id"]:
                self._objects[index] = validated
                return
        raise BuilderOperationError(f"object {validated['id']!r} does not exist")

    def _update_linked_camera_for_object(self, obj: dict[str, Any]) -> None:
        x, y, z = [float(value) for value in obj["position"]]
        if obj["semantic_type"] == "agent":
            self._update_agent_camera(x, y, z)
        elif obj["semantic_type"] == "goal":
            self._update_goal_camera(x, y, z)

    def _finalization_issues(self) -> list[str]:
        issues: list[str] = []
        if not self._objects:
            issues.append("add at least one scene object")
        semantics = {str(obj.get("semantic_type") or "") for obj in self._objects}
        if self.schema_version == "1.1" and self.game is None and {"agent", "goal"}.issubset(semantics):
            issues.append("scenes with an agent and goal require a reach-goal game contract")
        return issues

    def _default_cameras(self) -> list[dict[str, Any]]:
        width, depth, height = self.world_size
        return [
            {
                "id": "overview",
                "label": "Overview",
                "position": [0.0, -depth * 0.95, height * 0.9],
                "target": [0.0, 0.0, 0.0],
                "fov": 45.0,
            },
            {
                "id": "agent",
                "label": "Agent",
                "position": [-6.0, -7.0, 3.0],
                "target": [-4.0, 0.0, 0.8],
                "fov": 50.0,
            },
            {
                "id": "goal",
                "label": "Goal",
                "position": [6.0, -7.0, 3.0],
                "target": [4.0, 0.0, 0.8],
                "fov": 50.0,
            },
        ]

    def _update_agent_camera(self, x: float, y: float, z: float) -> None:
        for camera in self._cameras:
            if camera["id"] == "agent":
                camera["position"] = [x - 3.0, y - 5.0, z + 2.2]
                camera["target"] = [x, y, z]

    def _update_goal_camera(self, x: float, y: float, z: float) -> None:
        for camera in self._cameras:
            if camera["id"] == "goal":
                camera["position"] = [x + 3.0, y - 5.0, z + 2.2]
                camera["target"] = [x, y, z]

    def _allocate_id(self, prefix: str, *, requested: str | None = None) -> str:
        if requested is not None:
            cleaned = requested.strip()
            self._validate_id(cleaned, "requested id")
            if cleaned in self._claimed_ids:
                raise BuilderOperationError(f"id {cleaned!r} is already in use")
            self._claimed_ids.add(cleaned)
            return cleaned
        while True:
            self._id_counters[prefix] += 1
            candidate = f"{prefix}_{self._id_counters[prefix]}"
            if candidate not in self._claimed_ids:
                self._claimed_ids.add(candidate)
                return candidate

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        if not re.fullmatch(r"^[A-Za-z0-9_-]+$", value):
            raise BuilderOperationError(f"{label} must contain only letters, numbers, underscores, or hyphens")

    @classmethod
    def _clean_id(cls, value: str, label: str) -> str:
        cleaned = str(value).strip()
        cls._validate_id(cleaned, label)
        return cleaned

    @staticmethod
    def _clean_text(value: str, label: str) -> str:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            raise BuilderOperationError(f"{label} cannot be blank")
        return cleaned

    @staticmethod
    def _number(value: Number, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise BuilderOperationError(f"{label} must be a finite number")
        return float(value)

    @classmethod
    def _positive(cls, value: Number, label: str) -> float:
        number = cls._number(value, label)
        if number <= 0:
            raise BuilderOperationError(f"{label} must be positive")
        return number

    @classmethod
    def _vec3(cls, value: Vec3Input, label: str) -> list[float]:
        if len(value) != 3:
            raise BuilderOperationError(f"{label} must have exactly 3 values")
        return [cls._number(item, f"{label}[{index}]") for index, item in enumerate(value)]
