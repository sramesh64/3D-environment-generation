from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from typing import TYPE_CHECKING, Any

from .schema import Difficulty, EnvSpec3D, LevelFamily

if TYPE_CHECKING:
    from .builder import EnvSpec3DBuilder


COURTYARD_WIDTH = 24.0
COURTYARD_DEPTH = 18.0
COURTYARD_HEIGHT = 8.0
WORLD_SIZE = [30.0, 24.0, 10.0]
BOUNDARY_HEIGHT = 1.6
GENERATOR_VERSION = "courtyard_v1"
LEVEL_FAMILIES = {"barrier_route", "slalom", "push_lane", "elevation", "switch_gate", "mixed"}
DIFFICULTIES = {"easy", "medium", "hard"}


class CourtyardGenerationError(ValueError):
    """Raised when no valid deterministic courtyard can be generated."""


def courtyard_shell_spec(
    env_id: str,
    *,
    description: str = "Empty robot courtyard.",
    seed: int | None = None,
) -> EnvSpec3D:
    from .builder import EnvSpec3DBuilder

    builder = EnvSpec3DBuilder(env_id, description=description)
    populate_courtyard_shell(builder, seed=seed)
    return builder.finalize()


def populate_courtyard_shell(builder: EnvSpec3DBuilder, *, seed: int | None = None) -> None:
    if builder._objects:
        raise CourtyardGenerationError("courtyard shell must be created before other objects")
    if seed is not None:
        _valid_seed(seed)
    # Courtyard shells are new authored scenes, so retain their typed appearance
    # fields across the blank-before snapshot and subsequent edits.
    builder.schema_version = "1.1"
    builder.set_world(width=WORLD_SIZE[0], depth=WORLD_SIZE[1], height=WORLD_SIZE[2], theme="robot_courtyard")
    builder.theme = "robot_courtyard"
    _add_shell(builder)


def is_courtyard_shell(spec: EnvSpec3D | dict[str, Any]) -> bool:
    try:
        parsed = spec if isinstance(spec, EnvSpec3D) else EnvSpec3D.model_validate(spec)
    except ValueError:
        return False
    expected_ids = {"ground", "north_boundary", "south_boundary", "west_boundary", "east_boundary"}
    return (
        parsed.theme == "robot_courtyard"
        and parsed.game is None
        and not parsed.generation
        and not parsed.mechanisms
        and {obj.id for obj in parsed.objects} == expected_ids
        and len(parsed.objects) == len(expected_ids)
    )


def populate_courtyard_level(
    builder: EnvSpec3DBuilder,
    *,
    family: str,
    difficulty: str,
    seed: int | None,
) -> None:
    family_name = _choice(family, LEVEL_FAMILIES, "family")
    difficulty_name = _choice(difficulty, DIFFICULTIES, "difficulty")
    base_seed = _seed(builder.env_id) if seed is None else _valid_seed(seed)
    last_issues: list[str] = []
    for attempt in range(20):
        candidate = type(builder)(builder.env_id, description=builder.description)
        rng = random.Random(_seed(str(base_seed), str(attempt), family_name, difficulty_name))
        _build_level(candidate, family_name, difficulty_name, base_seed, attempt, rng)
        spec = candidate.finalize()
        last_issues = validate_courtyard_layout(spec)
        if not last_issues:
            builder.__dict__.update(candidate.__dict__)
            return
    raise CourtyardGenerationError(
        f"could not generate a valid {difficulty_name} {family_name} courtyard after 20 attempts: "
        + "; ".join(last_issues)
    )


def validate_courtyard_layout(spec: EnvSpec3D | dict[str, Any]) -> list[str]:
    parsed = spec if isinstance(spec, EnvSpec3D) else EnvSpec3D.model_validate(spec)
    issues: list[str] = []
    if parsed.theme != "robot_courtyard":
        issues.append("courtyard theme must be robot_courtyard")
    if parsed.game is None:
        return [*issues, "courtyard level is missing its game contract"]
    objects = {obj.id: obj for obj in parsed.objects}
    agent = objects[parsed.game.agent_id]
    goal = objects[parsed.game.goal_id]
    distance = math.hypot(agent.position[0] - goal.position[0], agent.position[1] - goal.position[1])
    if distance < 10.0:
        issues.append(f"agent and goal must be at least 10 m apart; found {distance:.2f} m")
    half_width = parsed.game.play_bounds[0] * 0.5
    half_depth = parsed.game.play_bounds[1] * 0.5
    for obj in parsed.objects:
        if obj.semantic_type in {"ground", "wall"} and "boundary" in obj.tags:
            continue
        if abs(obj.position[0]) > half_width or abs(obj.position[1]) > half_depth:
            issues.append(f"object {obj.id!r} lies outside the courtyard play bounds")
    for obj in parsed.objects:
        if obj.semantic_type == "ramp":
            rise = float(obj.metadata.get("rise", obj.metadata.get("height", 0.0)))
            slope = math.degrees(math.atan2(rise, obj.size[0]))
            if slope > 30.0:
                issues.append(f"ramp {obj.id!r} slope {slope:.1f} degrees exceeds 30 degrees")
    for obj in parsed.objects:
        if obj.id == agent.id or obj.semantic_type in {"ground", "platform", "ramp", "goal", "floor_switch"}:
            continue
        if _footprints_overlap(agent, obj, margin=0.12):
            issues.append(f"agent spawn overlaps {obj.id!r}")
    for obj in parsed.objects:
        if obj.id == goal.id or obj.semantic_type in {"ground", "platform", "ramp", "agent", "floor_switch"}:
            continue
        if _footprints_overlap(goal, obj, margin=0.08):
            issues.append(f"goal zone overlaps {obj.id!r}")
    if not _has_grid_path(parsed, agent.position, goal.position, gates_blocked=False):
        issues.append("no conservative ground route exists from the agent to the goal")
    for mechanism in parsed.mechanisms:
        trigger = objects[mechanism.trigger_id]
        if not _has_grid_path(parsed, agent.position, trigger.position, gates_blocked=True):
            issues.append(f"floor switch {trigger.id!r} is not reachable while its gate is closed")
        if not _has_grid_path(parsed, trigger.position, goal.position, gates_blocked=False):
            issues.append(f"goal is not reachable after mechanism {mechanism.id!r} opens")
    return issues


def _build_level(
    builder: EnvSpec3DBuilder,
    family: LevelFamily,
    difficulty: Difficulty,
    seed: int,
    attempt: int,
    rng: random.Random,
) -> None:
    builder.schema_version = "1.1"
    builder.set_world(width=WORLD_SIZE[0], depth=WORLD_SIZE[1], height=WORLD_SIZE[2], theme="robot_courtyard")
    builder.theme = "robot_courtyard"
    builder.generation = {
        "seed": seed,
        "family": family,
        "difficulty": difficulty,
        "generator_version": GENERATOR_VERSION,
        "attempt": attempt,
    }
    _add_shell(builder)

    start = [-6.2, -4.2]
    goal_xy = [5.4, 3.4]
    goal_z = 0.0
    if family == "barrier_route":
        _barrier_route(builder, difficulty, rng)
    elif family == "slalom":
        _slalom(builder, difficulty, rng)
    elif family == "push_lane":
        _push_lane(builder, difficulty, rng)
    elif family == "elevation":
        goal_xy, goal_z = _elevation(builder, difficulty, rng)
    elif family == "switch_gate":
        _switch_gate(builder, difficulty, rng)
    else:
        _mixed(builder, difficulty, rng)

    agent_id = builder.add_agent_spawn(*start, id="agent")
    goal_id = builder.add_goal_zone(*goal_xy, z=goal_z, id="goal")
    builder.set_object_appearance(agent_id, "courtyard_robot")
    builder.set_object_appearance(goal_id, "courtyard_goal_pad")
    builder.game = {
        "mode": "reach_goal",
        "agent_id": agent_id,
        "goal_id": goal_id,
        "play_bounds": [COURTYARD_WIDTH, COURTYARD_DEPTH, COURTYARD_HEIGHT],
        "reset_on": ["out_of_bounds"],
    }


def _add_shell(builder: EnvSpec3DBuilder) -> None:
    ground = builder.add_ground_plane(COURTYARD_WIDTH, COURTYARD_DEPTH, id="ground")
    builder.set_object_appearance(ground, "courtyard_ground", "grass_pavers")
    walls = [
        builder.add_wall(0, COURTYARD_DEPTH / 2, width=COURTYARD_WIDTH, depth=0.3, height=BOUNDARY_HEIGHT, id="north_boundary"),
        builder.add_wall(0, -COURTYARD_DEPTH / 2, width=COURTYARD_WIDTH, depth=0.3, height=BOUNDARY_HEIGHT, id="south_boundary"),
        builder.add_wall(-COURTYARD_WIDTH / 2, 0, width=0.3, depth=COURTYARD_DEPTH, height=BOUNDARY_HEIGHT, id="west_boundary"),
        builder.add_wall(COURTYARD_WIDTH / 2, 0, width=0.3, depth=COURTYARD_DEPTH, height=BOUNDARY_HEIGHT, id="east_boundary"),
    ]
    for wall_id in walls:
        wall = builder._object_by_id(wall_id)
        wall["tags"] = [*wall["tags"], "boundary"]
        builder._replace_object(wall)
        builder.set_object_appearance(wall_id, "courtyard_boundary", "plain")


def _barrier_route(builder: EnvSpec3DBuilder, difficulty: Difficulty, rng: random.Random) -> None:
    offset = rng.uniform(-0.5, 0.5)
    wall = builder.add_wall(offset, 0, width=0.4, depth=8.0, height=1.5, id="route_barrier")
    builder.set_object_appearance(wall, "courtyard_boundary", rng.choice(["hedge", "stone"]))
    if difficulty != "easy":
        _static_prop(builder, -3.2, 2.0, "planter", "route_planter")
    if difficulty == "hard":
        _hazard(builder, 3.2, -2.5, 2.4, 1.4, "broken_paving", "route_hazard")


def _slalom(builder: EnvSpec3DBuilder, difficulty: Difficulty, rng: random.Random) -> None:
    count = {"easy": 4, "medium": 6, "hard": 8}[difficulty]
    for index in range(count):
        x = -4.8 + index * (9.6 / max(1, count - 1))
        y = (2.1 if index % 2 == 0 else -1.6) + rng.uniform(-0.25, 0.25)
        _static_prop(builder, x, y, "planter" if index % 3 else "bench", f"slalom_prop_{index + 1}")
    if difficulty != "easy":
        _hazard(builder, 1.2, -4.2, 2.0, 1.3, "puddle", "slalom_hazard")


def _push_lane(builder: EnvSpec3DBuilder, difficulty: Difficulty, rng: random.Random) -> None:
    for y, name in ((-1.8, "lane_south"), (1.8, "lane_north")):
        wall = builder.add_wall(0, y, width=10.0, depth=0.3, height=1.2, id=name)
        builder.set_object_appearance(wall, "courtyard_boundary", "fence")
    crate = builder.add_pushable_box(-0.2, 0, width=1.2, depth=1.2, height=1.1, id="lane_crate")
    builder.set_object_appearance(crate, "courtyard_pushable_crate")
    if difficulty == "hard":
        barrel = builder.add_cylinder(2.1, 0.7, radius=0.42, height=1.0, id="lane_barrel")
        builder.set_object_appearance(barrel, "courtyard_barrel")


def _elevation(
    builder: EnvSpec3DBuilder,
    difficulty: Difficulty,
    rng: random.Random,
) -> tuple[list[float], float]:
    height = {"easy": 0.65, "medium": 0.85, "hard": 1.0}[difficulty]
    ramp = builder.add_ramp(2.0, 3.6, length=3.4, width=2.2, rise=height, id="goal_ramp")
    platform = builder.add_platform(6.3, 3.6, z=height, width=5.2, depth=3.2, id="goal_platform")
    builder.set_object_appearance(ramp, "courtyard_ramp", "wood")
    builder.set_object_appearance(platform, "courtyard_platform", "wood")
    if difficulty == "hard":
        _hazard(builder, 0.2, 3.6, 1.6, 2.4, "flowerbed", "ramp_hazard")
    return [6.7, 3.6], height + 0.3


def _switch_gate(builder: EnvSpec3DBuilder, difficulty: Difficulty, rng: random.Random) -> None:
    gap = 2.4
    segment_depth = (COURTYARD_DEPTH - gap) * 0.5
    for y, name in ((-(gap + segment_depth) * 0.5, "gate_wall_south"), ((gap + segment_depth) * 0.5, "gate_wall_north")):
        wall = builder.add_wall(0.7, y, width=0.35, depth=segment_depth, height=1.5, id=name)
        builder.set_object_appearance(wall, "courtyard_boundary", "stone")
    gate = builder.add_sliding_gate(0.7, 0, width=0.35, depth=gap - 0.25, height=1.8, id="main_gate")
    switch_y = 3.7 if rng.random() > 0.5 else -3.2
    switch = builder.add_floor_switch(-3.2, switch_y, id="gate_switch")
    builder.link_switch_to_gate(switch, gate, id="courtyard_gate_link")
    if difficulty == "hard":
        crate = builder.add_pushable_box(-4.6, switch_y, width=1.0, depth=1.0, height=0.9, id="switch_crate")
        builder.set_object_appearance(crate, "courtyard_pushable_crate")


def _mixed(builder: EnvSpec3DBuilder, difficulty: Difficulty, rng: random.Random) -> None:
    _barrier_route(builder, "easy" if difficulty == "easy" else "medium", rng)
    for index, (x, y) in enumerate([(-4.8, -0.6), (3.7, 2.7), (5.0, -3.2)]):
        if difficulty == "easy" and index > 0:
            break
        _static_prop(builder, x, y, "bench" if index == 1 else "planter", f"mixed_prop_{index + 1}")
    if difficulty != "easy":
        crate = builder.add_pushable_box(-2.8, -3.3, width=1.0, depth=1.0, height=1.0, id="mixed_crate")
        builder.set_object_appearance(crate, "courtyard_pushable_crate")


def _static_prop(builder: EnvSpec3DBuilder, x: float, y: float, variant: str, object_id: str) -> None:
    sizes = {"planter": (1.1, 1.1, 0.8), "bench": (1.8, 0.65, 0.85), "crate": (1.0, 1.0, 1.0)}
    width, depth, height = sizes[variant]
    item = builder.add_static_box(x, y, width=width, depth=depth, height=height, id=object_id)
    builder.set_object_appearance(item, "courtyard_static_prop", variant)


def _hazard(
    builder: EnvSpec3DBuilder,
    x: float,
    y: float,
    width: float,
    depth: float,
    variant: str,
    object_id: str,
) -> None:
    item = builder.add_hazard_zone(x, y, width=width, depth=depth, id=object_id)
    builder.set_object_appearance(item, "courtyard_hazard", variant)


def _has_grid_path(
    spec: EnvSpec3D,
    start: list[float] | tuple[float, ...],
    end: list[float] | tuple[float, ...],
    *,
    gates_blocked: bool,
) -> bool:
    resolution = 0.35
    half_width = COURTYARD_WIDTH * 0.5 - 0.55
    half_depth = COURTYARD_DEPTH * 0.5 - 0.55
    columns = int(math.floor((half_width * 2) / resolution)) + 1
    rows = int(math.floor((half_depth * 2) / resolution)) + 1

    def cell(position: list[float] | tuple[float, ...]) -> tuple[int, int]:
        return (
            max(0, min(columns - 1, int(round((float(position[0]) + half_width) / resolution)))),
            max(0, min(rows - 1, int(round((float(position[1]) + half_depth) / resolution)))),
        )

    blocked: set[tuple[int, int]] = set()
    clearance = 0.45
    for obj in spec.objects:
        if obj.semantic_type not in {"wall", "static_box", "gate"}:
            continue
        if obj.semantic_type == "wall" and "boundary" in obj.tags:
            continue
        if obj.semantic_type == "gate" and not gates_blocked:
            continue
        half_x, half_y = _yaw_aabb_half_extents(obj.size[0], obj.size[1], obj.yaw)
        min_x = float(obj.position[0]) - half_x - clearance
        max_x = float(obj.position[0]) + half_x + clearance
        min_y = float(obj.position[1]) - half_y - clearance
        max_y = float(obj.position[1]) + half_y + clearance
        for ix in range(columns):
            x = -half_width + ix * resolution
            if x < min_x or x > max_x:
                continue
            for iy in range(rows):
                y = -half_depth + iy * resolution
                if min_y <= y <= max_y:
                    blocked.add((ix, iy))
    start_cell = cell(start)
    end_cell = cell(end)
    blocked.discard(start_cell)
    blocked.discard(end_cell)
    queue = deque([start_cell])
    visited = {start_cell}
    while queue:
        current = queue.popleft()
        if current == end_cell:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = (current[0] + dx, current[1] + dy)
            if not (0 <= candidate[0] < columns and 0 <= candidate[1] < rows):
                continue
            if candidate in blocked or candidate in visited:
                continue
            visited.add(candidate)
            queue.append(candidate)
    return False


def _yaw_aabb_half_extents(width: float, depth: float, yaw: float) -> tuple[float, float]:
    cosine = abs(math.cos(yaw))
    sine = abs(math.sin(yaw))
    return (
        cosine * width * 0.5 + sine * depth * 0.5,
        sine * width * 0.5 + cosine * depth * 0.5,
    )


def _footprints_overlap(left: Any, right: Any, *, margin: float) -> bool:
    left_half = _yaw_aabb_half_extents(float(left.size[0]), float(left.size[1]), float(left.yaw))
    right_half = _yaw_aabb_half_extents(float(right.size[0]), float(right.size[1]), float(right.yaw))
    return (
        abs(float(left.position[0]) - float(right.position[0])) < left_half[0] + right_half[0] + margin
        and abs(float(left.position[1]) - float(right.position[1])) < left_half[1] + right_half[1] + margin
    )


def _choice(value: str, allowed: set[str], label: str) -> str:
    cleaned = str(value).strip().lower()
    if cleaned not in allowed:
        raise CourtyardGenerationError(f"unsupported {label} {value!r}; choose one of {sorted(allowed)}")
    return cleaned


def _valid_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise CourtyardGenerationError("seed must be an integer between 0 and 2^63-1")
    return value


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)
