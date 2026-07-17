from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Color = str
BodyType = Literal["static", "dynamic", "sensor", "mechanism"]
ShapeType = Literal["box", "sphere", "cylinder", "capsule", "ramp"]
LevelFamily = Literal["barrier_route", "slalom", "push_lane", "elevation", "switch_gate", "mixed"]
Difficulty = Literal["easy", "medium", "hard"]

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

COURTYARD_APPEARANCES: dict[str, tuple[set[str], set[str] | None]] = {
    "courtyard_ground": ({"ground"}, {"grass_pavers", "pavers_grass"}),
    "courtyard_boundary": ({"wall"}, {"fence", "hedge", "stone", "plain"}),
    "courtyard_static_prop": ({"static_box"}, {"planter", "bench", "crate"}),
    "courtyard_pushable_crate": ({"pushable_box"}, None),
    "courtyard_barrel": ({"cylinder"}, None),
    "courtyard_floor_switch": ({"floor_switch"}, None),
    "courtyard_platform": ({"platform"}, {"stone", "wood"}),
    "courtyard_ramp": ({"ramp"}, {"stone", "wood"}),
    "courtyard_gate": ({"gate"}, None),
    "courtyard_robot": ({"agent"}, None),
    "courtyard_goal_pad": ({"goal"}, None),
    "courtyard_target_region": ({"target_region"}, None),
    "courtyard_hazard": ({"hazard"}, {"puddle", "flowerbed", "broken_paving"}),
}


class SpecValidationError(ValueError):
    """Raised when an environment spec fails validation."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraSpec(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    position: Vec3
    target: Vec3
    fov: float = Field(default=45.0, gt=1.0, lt=160.0)

    @model_validator(mode="after")
    def validate_id(self) -> "CameraSpec":
        _validate_id(self.id, "camera id")
        return self


class AppearanceSpec(StrictModel):
    asset_id: str = Field(min_length=1)
    variant: str | None = None

    @model_validator(mode="after")
    def validate_asset_id(self) -> "AppearanceSpec":
        _validate_id(self.asset_id, "appearance asset_id")
        if self.variant is not None:
            _validate_id(self.variant, "appearance variant")
        return self


class GameSpec(StrictModel):
    mode: Literal["reach_goal"] = "reach_goal"
    agent_id: str = Field(min_length=1)
    goal_id: str = Field(min_length=1)
    play_bounds: Vec3 = [24.0, 18.0, 8.0]
    reset_on: list[Literal["hazard", "out_of_bounds"]] = Field(
        default_factory=lambda: ["out_of_bounds"]
    )

    @model_validator(mode="after")
    def validate_game(self) -> "GameSpec":
        _validate_id(self.agent_id, "game agent_id")
        _validate_id(self.goal_id, "game goal_id")
        if any(value <= 0 for value in self.play_bounds):
            raise ValueError("game play_bounds values must be positive")
        if len(self.reset_on) != len(set(self.reset_on)):
            raise ValueError("game reset_on entries must be unique")
        return self


class GenerationSpec(StrictModel):
    seed: int = Field(ge=0, le=2**63 - 1)
    family: LevelFamily
    difficulty: Difficulty = "medium"
    generator_version: Literal["courtyard_v1"] = "courtyard_v1"
    attempt: int = Field(default=0, ge=0, le=19)


class MechanismSpec(StrictModel):
    id: str = Field(min_length=1)
    trigger_id: str = Field(min_length=1)
    gate_id: str = Field(min_length=1)
    mode: Literal["latch_open"] = "latch_open"

    @model_validator(mode="after")
    def validate_mechanism(self) -> "MechanismSpec":
        _validate_id(self.id, "mechanism id")
        _validate_id(self.trigger_id, "mechanism trigger_id")
        _validate_id(self.gate_id, "mechanism gate_id")
        return self


class EnvObject3D(StrictModel):
    id: str = Field(min_length=1)
    semantic_type: str = Field(min_length=1)
    shape: ShapeType
    body_type: BodyType
    position: Vec3
    size: Vec3
    yaw: float = 0.0
    label: str | None = None
    tags: list[str] = Field(default_factory=list)
    color: Color | None = None
    visible: bool = True
    appearance: AppearanceSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_object(self) -> "EnvObject3D":
        _validate_id(self.id, "object id")
        if any(value <= 0 for value in self.size):
            raise ValueError("object size values must be positive full dimensions")
        if self.shape == "sphere" and not _same(self.size[0], self.size[1], self.size[2]):
            raise ValueError("sphere size must use equal x, y, and z dimensions")
        if self.shape in {"cylinder", "capsule"} and not _same(self.size[0], self.size[1]):
            raise ValueError(f"{self.shape} size x and y must match the diameter")
        if self.shape == "ramp":
            if self.semantic_type != "ramp" or self.body_type != "static":
                raise ValueError("ramp shapes must use semantic_type 'ramp' and body_type 'static'")
            if self.metadata.get("geometry_version") == 2:
                if "rise" not in self.metadata:
                    raise ValueError("canonical ramp metadata requires rise")
                low_end = self.metadata.get("low_end")
                if not isinstance(low_end, list) or len(low_end) != 3:
                    raise ValueError("canonical ramp metadata requires a three-coordinate low_end")
                from .ramp_geometry import ramp_geometry_from_object

                geometry = ramp_geometry_from_object(self.model_dump(mode="json"))
                if any(abs(actual - expected) > 1e-6 for actual, expected in zip(self.position, geometry.center)):
                    raise ValueError("canonical ramp position is inconsistent with low_end, run, rise, and yaw")
        if self.body_type == "sensor" and self.shape not in {"box", "sphere", "cylinder"}:
            raise ValueError("sensor objects must use box, sphere, or cylinder shapes")
        if self.body_type == "mechanism" and self.semantic_type != "gate":
            raise ValueError("mechanism bodies currently support only gate objects")
        if self.semantic_type == "gate" and self.body_type != "mechanism":
            raise ValueError("gate objects must use the mechanism body type")
        if self.semantic_type == "floor_switch" and self.body_type != "sensor":
            raise ValueError("floor_switch objects must use the sensor body type")
        if self.color is not None:
            _validate_color(self.color)
        if self.appearance is not None:
            rule = COURTYARD_APPEARANCES.get(self.appearance.asset_id)
            if rule is None:
                raise ValueError(f"unsupported appearance asset_id {self.appearance.asset_id!r}")
            semantics, variants = rule
            if self.semantic_type not in semantics:
                raise ValueError(
                    f"appearance {self.appearance.asset_id!r} is not valid for semantic_type {self.semantic_type!r}"
                )
            if variants is not None and self.appearance.variant not in variants:
                raise ValueError(
                    f"appearance {self.appearance.asset_id!r} requires variant in {sorted(variants)}"
                )
            if variants is None and self.appearance.variant is not None:
                raise ValueError(f"appearance {self.appearance.asset_id!r} does not accept a variant")
        return self


class EnvSpec3D(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    world_size: Vec3 = [24.0, 18.0, 8.0]
    gravity: Vec3 = [0.0, 0.0, -9.81]
    theme: str = "storybook_adventure"
    cameras: list[CameraSpec] = Field(default_factory=list)
    objects: list[EnvObject3D] = Field(default_factory=list)
    game: GameSpec | None = None
    generation: GenerationSpec | None = None
    mechanisms: list[MechanismSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_spec(self) -> "EnvSpec3D":
        _validate_id(self.id, "environment id")
        if any(value <= 0 for value in self.world_size):
            raise ValueError("world_size values must be positive")
        object_ids = [obj.id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object ids must be unique")
        camera_ids = [camera.id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera ids must be unique")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        if self.schema_version == "1.0" and (self.game or self.generation or self.mechanisms):
            raise ValueError("game, generation, and mechanisms require schema_version 1.1")
        objects = {obj.id: obj for obj in self.objects}
        if self.game:
            agent = objects.get(self.game.agent_id)
            goal = objects.get(self.game.goal_id)
            if agent is None or agent.semantic_type != "agent":
                raise ValueError("game agent_id must reference an agent object")
            if goal is None or goal.semantic_type != "goal":
                raise ValueError("game goal_id must reference a goal object")
            if sum(obj.semantic_type == "agent" for obj in self.objects) != 1:
                raise ValueError("game scenes must contain exactly one agent")
            if sum(obj.semantic_type == "goal" for obj in self.objects) != 1:
                raise ValueError("game scenes must contain exactly one goal")
        mechanism_ids = [mechanism.id for mechanism in self.mechanisms]
        if len(mechanism_ids) != len(set(mechanism_ids)):
            raise ValueError("mechanism ids must be unique")
        for mechanism in self.mechanisms:
            trigger = objects.get(mechanism.trigger_id)
            gate = objects.get(mechanism.gate_id)
            if trigger is None or trigger.semantic_type != "floor_switch":
                raise ValueError(f"mechanism {mechanism.id!r} trigger_id must reference a floor_switch")
            if gate is None or gate.semantic_type != "gate":
                raise ValueError(f"mechanism {mechanism.id!r} gate_id must reference a gate")
        return self


def parse_env_spec_3d(value: dict[str, Any]) -> EnvSpec3D:
    try:
        return EnvSpec3D.model_validate(value)
    except ValidationError as exc:
        raise SpecValidationError(str(exc)) from exc


def env_spec_to_dict(spec: EnvSpec3D) -> dict[str, Any]:
    """Serialize without introducing v1.1 defaults into legacy v1.0 specs."""

    value = spec.model_dump(mode="json", exclude_none=True)
    if spec.schema_version == "1.0":
        value.pop("game", None)
        value.pop("generation", None)
        value.pop("mechanisms", None)
        for obj in value.get("objects") or []:
            obj.pop("appearance", None)
    return value


def validate_env_spec_3d(value: dict[str, Any] | EnvSpec3D) -> dict[str, Any]:
    spec = value if isinstance(value, EnvSpec3D) else parse_env_spec_3d(value)
    return env_spec_to_dict(spec)


def _validate_id(value: str, field_name: str) -> None:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} must match {_ID_RE.pattern}")


def _validate_color(value: str) -> None:
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?", value):
        raise ValueError("colors must be #RRGGBB or #RRGGBBAA")


def _same(*values: float) -> bool:
    first = values[0]
    return all(abs(value - first) <= 1e-9 for value in values[1:])
