from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .builder import BuilderOperationError, EnvSpec3DBuilder


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetWorldArgs(StrictModel):
    width: float = Field(default=24, gt=0)
    depth: float = Field(default=18, gt=0)
    height: float = Field(default=8, gt=0)
    gravity: list[float] | None = Field(default=None, min_length=3, max_length=3)
    theme: str | None = None


class SetDescriptionArgs(StrictModel):
    description: str = Field(min_length=1)


class MoveObjectArgs(StrictModel):
    id: str = Field(min_length=1)
    x: float | None = None
    y: float | None = None
    z: float | None = None


class RotateObjectArgs(StrictModel):
    id: str = Field(min_length=1)
    yaw: float


class SetObjectAppearanceArgs(StrictModel):
    id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    variant: str | None = None


class ResizeObjectArgs(StrictModel):
    id: str = Field(min_length=1)
    width: float | None = Field(default=None, gt=0)
    depth: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)


class RemoveObjectArgs(StrictModel):
    id: str = Field(min_length=1)


class AddGroundPlaneArgs(StrictModel):
    width: float | None = Field(default=None, gt=0)
    depth: float | None = Field(default=None, gt=0)
    thickness: float = Field(default=0.2, gt=0)
    id: str | None = None


class AddWallArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=0.4, gt=0)
    depth: float = Field(default=6, gt=0)
    height: float = Field(default=2.5, gt=0)
    yaw: float = 0
    id: str | None = None


class AddPlatformArgs(StrictModel):
    x: float
    y: float
    z: float
    width: float = Field(default=4, gt=0)
    depth: float = Field(default=4, gt=0)
    thickness: float = Field(default=0.3, gt=0)
    yaw: float = 0
    id: str | None = None


class AddRampArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    length: float = Field(default=4, gt=0)
    width: float = Field(default=2, gt=0)
    rise: float = Field(default=1.5, gt=0)
    thickness: float = Field(default=0.25, gt=0)
    yaw: float = 0
    id: str | None = None


class SetRampGeometryArgs(StrictModel):
    id: str = Field(min_length=1)
    x: float | None = None
    y: float | None = None
    z: float | None = None
    length: float | None = Field(default=None, gt=0)
    width: float | None = Field(default=None, gt=0)
    rise: float | None = Field(default=None, gt=0)
    thickness: float | None = Field(default=None, gt=0)
    yaw: float | None = None

    @model_validator(mode="after")
    def require_geometry_change(self) -> "SetRampGeometryArgs":
        if all(
            value is None
            for value in (self.x, self.y, self.z, self.length, self.width, self.rise, self.thickness, self.yaw)
        ):
            raise ValueError("set_ramp_geometry requires at least one geometry field")
        return self


class AddPushableBoxArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=1, gt=0)
    depth: float = Field(default=1, gt=0)
    height: float = Field(default=1, gt=0)
    yaw: float = 0
    id: str | None = None


class AddStaticBoxArgs(AddPushableBoxArgs):
    pass


class AddBallArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    radius: float = Field(default=0.5, gt=0)
    id: str | None = None


class AddCylinderArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    radius: float = Field(default=0.45, gt=0)
    height: float = Field(default=1, gt=0)
    id: str | None = None


class AddAgentSpawnArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    radius: float = Field(default=0.35, gt=0)
    height: float = Field(default=1.1, gt=0)
    id: str | None = None


class AddGoalZoneArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=1.4, gt=0)
    depth: float = Field(default=1.4, gt=0)
    height: float = Field(default=1.2, gt=0)
    id: str | None = None


class AddHazardZoneArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=2, gt=0)
    depth: float = Field(default=2, gt=0)
    height: float = Field(default=0.15, gt=0)
    id: str | None = None


class AddTargetRegionArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=1.8, gt=0)
    depth: float = Field(default=1.8, gt=0)
    height: float = Field(default=0.2, gt=0)
    id: str | None = None


class AddFloorSwitchArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=1.2, gt=0)
    depth: float = Field(default=1.2, gt=0)
    height: float = Field(default=0.12, gt=0)
    id: str | None = None


class AddSlidingGateArgs(StrictModel):
    x: float
    y: float
    z: float = 0
    width: float = Field(default=3.0, gt=0)
    depth: float = Field(default=0.3, gt=0)
    height: float = Field(default=1.8, gt=0)
    yaw: float = 0
    travel: float | None = Field(default=None, gt=0)
    id: str | None = None


class LinkSwitchToGateArgs(StrictModel):
    trigger_id: str = Field(min_length=1)
    gate_id: str = Field(min_length=1)
    id: str | None = None


class ConfigureReachGoalGameArgs(StrictModel):
    agent_id: str = Field(min_length=1)
    goal_id: str = Field(min_length=1)


class MakeCourtyardLevelArgs(StrictModel):
    family: Literal["barrier_route", "slalom", "push_lane", "elevation", "switch_gate", "mixed"] = "barrier_route"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class MakeEmptyRoomArgs(StrictModel):
    width: float = Field(default=16, gt=0)
    depth: float = Field(default=12, gt=0)
    wall_height: float = Field(default=1.2, gt=0)


class EmptyArgs(StrictModel):
    pass


OPERATION_ARG_MODELS: dict[str, type[BaseModel]] = {
    "set_world": SetWorldArgs,
    "set_description": SetDescriptionArgs,
    "move_object": MoveObjectArgs,
    "rotate_object": RotateObjectArgs,
    "set_object_appearance": SetObjectAppearanceArgs,
    "resize_object": ResizeObjectArgs,
    "remove_object": RemoveObjectArgs,
    "add_ground_plane": AddGroundPlaneArgs,
    "add_wall": AddWallArgs,
    "add_platform": AddPlatformArgs,
    "add_ramp": AddRampArgs,
    "set_ramp_geometry": SetRampGeometryArgs,
    "add_static_box": AddStaticBoxArgs,
    "add_pushable_box": AddPushableBoxArgs,
    "add_ball": AddBallArgs,
    "add_cylinder": AddCylinderArgs,
    "add_agent_spawn": AddAgentSpawnArgs,
    "add_goal_zone": AddGoalZoneArgs,
    "add_hazard_zone": AddHazardZoneArgs,
    "add_target_region": AddTargetRegionArgs,
    "add_floor_switch": AddFloorSwitchArgs,
    "add_sliding_gate": AddSlidingGateArgs,
    "link_switch_to_gate": LinkSwitchToGateArgs,
    "configure_reach_goal_game": ConfigureReachGoalGameArgs,
    "make_courtyard_level": MakeCourtyardLevelArgs,
    "make_empty_room": MakeEmptyRoomArgs,
    "make_ramp_course": EmptyArgs,
    "make_box_goal_scene": EmptyArgs,
}

OPERATION_DESCRIPTIONS: dict[str, str] = {
    "set_world": "Configure an empty z-up 3D world before adding objects.",
    "set_description": "Replace the environment description without changing geometry.",
    "move_object": "Move a non-ramp object by id. Coordinates are object center coordinates.",
    "rotate_object": "Rotate a non-ramp object around the z axis using radians.",
    "set_object_appearance": "Choose a validated robot-courtyard visual asset and variant without changing physics.",
    "resize_object": "Resize a non-ramp object by id. Dimensions are full width, depth, and height.",
    "remove_object": "Remove an existing object by id.",
    "add_ground_plane": "Add a bounded floor slab.",
    "add_wall": "Add a static vertical barrier box.",
    "add_platform": "Add a static raised platform.",
    "add_ramp": (
        "Add a ramp from a low walkable endpoint. yaw points uphill; length is horizontal run; "
        "rise is vertical gain; thickness is collider thickness."
    ),
    "set_ramp_geometry": (
        "Edit a ramp using its low walkable endpoint. Omitted fields are preserved; yaw points uphill, "
        "length is horizontal run, rise is vertical gain, and thickness is collider thickness."
    ),
    "add_static_box": "Add an anchored crate-shaped box for stable structures.",
    "add_pushable_box": "Add a dynamic pushable box.",
    "add_ball": "Add a dynamic sphere.",
    "add_cylinder": "Add a dynamic upright cylinder.",
    "add_agent_spawn": "Add the visible agent spawn marker.",
    "add_goal_zone": "Add a visible non-colliding goal region.",
    "add_hazard_zone": "Add a visible non-colliding hazard region.",
    "add_target_region": "Add a neutral non-colliding delivery or task target region.",
    "add_floor_switch": "Add a non-colliding floor switch sensor.",
    "add_sliding_gate": "Add a vertically sliding courtyard gate mechanism body.",
    "link_switch_to_gate": "Link one floor switch to one latch-open sliding gate.",
    "configure_reach_goal_game": "Activate reach-goal gameplay using an already-authored agent and goal.",
    "make_courtyard_level": "Generate a deterministic playable robot courtyard from a family, difficulty, and seed.",
    "make_empty_room": "Create a floor and four low boundary walls.",
    "make_ramp_course": "Create a small ramp-to-goal baseline.",
    "make_box_goal_scene": "Create a box-and-wall goal baseline.",
}


class Operation(StrictModel):
    op: Literal[
        "set_world",
        "set_description",
        "move_object",
        "rotate_object",
        "set_object_appearance",
        "resize_object",
        "remove_object",
        "add_ground_plane",
        "add_wall",
        "add_platform",
        "add_ramp",
        "set_ramp_geometry",
        "add_static_box",
        "add_pushable_box",
        "add_ball",
        "add_cylinder",
        "add_agent_spawn",
        "add_goal_zone",
        "add_hazard_zone",
        "add_target_region",
        "add_floor_switch",
        "add_sliding_gate",
        "link_switch_to_gate",
        "configure_reach_goal_game",
        "make_courtyard_level",
        "make_empty_room",
        "make_ramp_course",
        "make_box_goal_scene",
    ]
    args: dict[str, Any] = Field(default_factory=dict)


def execute_operation(builder: EnvSpec3DBuilder, operation: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized_operation = _normalize_legacy_operation(operation)
        parsed = Operation.model_validate(normalized_operation)
        arg_model = OPERATION_ARG_MODELS[parsed.op]
        args = arg_model.model_validate(parsed.args).model_dump(exclude_none=True)
        result = getattr(builder, parsed.op)(**args)
    except (ValidationError, BuilderOperationError, AttributeError) as exc:
        return {
            "success": False,
            "operation": operation,
            "created_ids": [],
            "error": str(exc),
            "draft_summary": builder.inspect_draft(),
        }
    created_ids = [result] if isinstance(result, str) else []
    return {
        "success": True,
        "operation": {"op": parsed.op, "args": args},
        "created_ids": created_ids,
        "result": result,
        "draft_summary": builder.inspect_draft(),
    }


def builder_operation_arg_schemas() -> dict[str, dict[str, Any]]:
    return {
        name: model.model_json_schema()
        for name, model in OPERATION_ARG_MODELS.items()
    }


def _normalize_legacy_operation(operation: dict[str, Any]) -> dict[str, Any]:
    """Accept the old add_ramp height spelling while advertising only rise."""

    if not isinstance(operation, dict) or operation.get("op") != "add_ramp":
        return operation
    raw_args = operation.get("args")
    if not isinstance(raw_args, dict) or "height" not in raw_args or "rise" in raw_args:
        return operation
    args = dict(raw_args)
    args["rise"] = args.pop("height")
    return {**operation, "args": args}
