from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ENV_SPEC_FILENAME, WORLD_XML_FILENAME
from .runtime_config import configure_mujoco_gl
from .scene_geometry import volumes_overlap, yaw_from_rotation_matrix
from .schema import EnvObject3D, EnvSpec3D


MOVE_SPEED = 2.8
MOVE_ACCELERATION = 42.0
MOVE_DECELERATION = 64.0
AIR_CONTROL_FACTOR = 0.65
INPUT_GRACE_SECONDS = 0.35
JUMP_SPEED = 4.6
JUMP_BUFFER_SECONDS = 0.12
JUMP_COYOTE_SECONDS = 0.1
JUMP_COOLDOWN_SECONDS = 0.2
GROUND_NORMAL_Z_MIN = 0.55


@dataclass(frozen=True)
class AgentControl:
    object_id: str
    body_id: int
    geom_ids: frozenset[int]
    qpos_address: int
    qvel_address: int
    height: float


@dataclass
class MechanismControl:
    id: str
    trigger_id: str
    gate_id: str
    actuator_id: int
    joint_qpos_address: int
    travel: float
    latched: bool = False


@dataclass(frozen=True)
class _RuntimeShape:
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class _AgentContactState:
    grounded: bool
    blocking_normals: tuple[tuple[float, float], ...]
    ceiling_blocked: bool


@dataclass(frozen=True)
class RuntimeZoneEvent:
    sequence: int
    step: int
    simulation_time: float
    type: str
    subject_id: str
    zone_id: str
    semantic_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "step": self.step,
            "simulation_time": self.simulation_time,
            "type": self.type,
            "subject_id": self.subject_id,
            "zone_id": self.zone_id,
            "semantic_type": self.semantic_type,
        }


class PlayableSimulation:
    def __init__(self, *, model: Any, data: Any, spec: EnvSpec3D, mujoco_module: Any) -> None:
        self.model = model
        self.data = data
        self.spec = spec
        self.mujoco = mujoco_module
        self.agent = _find_agent_control(model, spec, mujoco_module)
        self.dynamic_body_ids = _find_dynamic_body_ids(model, spec, mujoco_module)
        self.sensor_zones = [obj for obj in spec.objects if obj.body_type == "sensor"]
        self.switch_zones = {obj.id: obj for obj in spec.objects if obj.semantic_type == "floor_switch"}
        self.object_specs = {obj.id: obj for obj in spec.objects}
        self.agent_spec = next(obj for obj in spec.objects if obj.id == self.agent.object_id)
        self._blocking_geom_ids = _find_blocking_geom_ids(model, spec, mujoco_module)
        self.mechanisms = _find_mechanism_controls(model, spec, mujoco_module)
        self._heading_yaw = next(obj.yaw for obj in spec.objects if obj.id == self.agent.object_id)
        self.reset()

    @classmethod
    def from_scene(cls, source: str | Path) -> "PlayableSimulation":
        configure_mujoco_gl()
        try:
            import mujoco
        except Exception as exc:  # pragma: no cover - dependency is environment-specific
            raise RuntimeError(f"MuJoCo is unavailable: {exc}") from exc

        scene_dir, xml_path, spec_path = resolve_scene_paths(source)
        del scene_dir
        spec = EnvSpec3D.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        return cls(model=model, data=data, spec=spec, mujoco_module=mujoco)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self._heading_yaw = next(obj.yaw for obj in self.spec.objects if obj.id == self.agent.object_id)
        self._jump_pressed = False
        self._jump_buffer_until = -math.inf
        self._last_grounded_at = -math.inf
        self._last_jump_at = -math.inf
        self.step_count = 0
        self.jump_count = 0
        self.distance_travelled = 0.0
        self.hazard_entries = 0
        self.goal_entries = 0
        self.goal_reached = False
        self._event_sequence = 0
        self._event_log: list[RuntimeZoneEvent] = []
        for mechanism in self.mechanisms:
            mechanism.latched = False
            self.data.ctrl[mechanism.actuator_id] = 0.0
        self._keep_agent_upright()
        self.mujoco.mj_forward(self.model, self.data)
        self._active_zone_pairs: set[tuple[str, str]] = set()
        self._record_zone_events(initial=True)
        if self.is_grounded():
            self._last_grounded_at = float(self.data.time)

    def step(self, *, right: float, forward: float, camera_azimuth: float, jump: bool = False) -> None:
        before_position = self.agent_position()
        desired_velocity = camera_relative_velocity(
            camera_azimuth,
            right=right,
            forward=forward,
            speed=MOVE_SPEED,
        )
        if math.hypot(*desired_velocity) > 1e-6:
            self._heading_yaw = math.atan2(desired_velocity[0], -desired_velocity[1])
        self._keep_agent_upright()
        qvel = self.agent.qvel_address
        contact_state = self._agent_contact_state()
        desired_velocity = _project_out_of_contacts(desired_velocity, contact_state.blocking_normals)
        current_velocity = (float(self.data.qvel[qvel]), float(self.data.qvel[qvel + 1]))
        acceleration = MOVE_ACCELERATION if math.hypot(*desired_velocity) > 1e-6 else MOVE_DECELERATION
        if not contact_state.grounded:
            acceleration *= AIR_CONTROL_FACTOR
        horizontal_velocity = _move_toward_velocity(
            current_velocity,
            desired_velocity,
            max_delta=acceleration * float(self.model.opt.timestep),
        )
        horizontal_velocity = _project_out_of_contacts(horizontal_velocity, contact_state.blocking_normals)
        self.data.qvel[qvel] = horizontal_velocity[0]
        self.data.qvel[qvel + 1] = horizontal_velocity[1]
        self.data.qvel[qvel + 3 : qvel + 6] = 0.0
        now = float(self.data.time)
        if contact_state.grounded:
            self._last_grounded_at = now
        if jump and not self._jump_pressed:
            self._jump_buffer_until = now + JUMP_BUFFER_SECONDS
        self._jump_pressed = jump
        can_use_ground = now - self._last_grounded_at <= JUMP_COYOTE_SECONDS
        cooldown_elapsed = now - self._last_jump_at >= JUMP_COOLDOWN_SECONDS
        if self._jump_buffer_until >= now and can_use_ground and cooldown_elapsed:
            self.data.qvel[qvel + 2] = JUMP_SPEED
            self.jump_count += 1
            self._last_jump_at = now
            self._last_grounded_at = -math.inf
            self._jump_buffer_until = -math.inf
        vertical_velocity_before_step = float(self.data.qvel[qvel + 2])
        self._update_mechanisms()
        self.mujoco.mj_step(self.model, self.data)
        contact_state = self._agent_contact_state()
        if not contact_state.grounded and not contact_state.ceiling_blocked:
            # A game character's airborne motion is ballistic. MuJoCo side
            # friction must not cancel gravity or a jump when contacts flicker
            # along walls, box sides, platform edges, or dynamic objects.
            self.data.qvel[qvel + 2] = vertical_velocity_before_step + (
                float(self.model.opt.gravity[2]) * float(self.model.opt.timestep)
            )
        self._update_mechanisms()
        after_position = self.agent_position()
        self.distance_travelled += math.dist(before_position[:2], after_position[:2])
        self.step_count += 1
        self._record_zone_events()
        if contact_state.grounded:
            self._last_grounded_at = float(self.data.time)

    def agent_position(self) -> tuple[float, float, float]:
        address = self.agent.qpos_address
        return tuple(float(value) for value in self.data.qpos[address : address + 3])  # type: ignore[return-value]

    def status(self) -> str:
        if self.active_zone_ids("hazard"):
            return "In hazard"
        if self.active_zone_ids("goal"):
            return "In goal"
        return "Exploring"

    def game_state(self, *, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        failure_reason = self.safety_failure_reason()
        return {
            "state": "failed" if failure_reason else "playing",
            "failure_reason": failure_reason,
            "events": list(events or []),
            "active_zones": {
                "goal": self.active_zone_ids("goal"),
                "hazard": self.active_zone_ids("hazard"),
            },
            "mechanisms": self.mechanism_states(),
            "metrics": {
                "steps": self.step_count,
                "elapsed_seconds": float(self.data.time),
                "distance_travelled": self.distance_travelled,
                "jumps": self.jump_count,
                "hazards": self.hazard_entries,
                "goals": self.goal_entries,
                "goal_reached": self.goal_reached,
            },
        }

    def safety_failure_reason(self) -> str:
        x, y, z = self.agent_position()
        bounds = self.spec.game.play_bounds if self.spec.game else self.spec.world_size
        half_x = float(bounds[0]) * 0.5
        half_y = float(bounds[1]) * 0.5
        if abs(x) > half_x + 0.5 or abs(y) > half_y + 0.5 or z < -1.0:
            return "out_of_bounds"
        return ""

    def failure_reason(self) -> str:
        """Backward-compatible alias for hard simulator safety failures."""

        return self.safety_failure_reason()

    def mechanism_states(self) -> list[dict[str, Any]]:
        return [
            {
                "id": mechanism.id,
                "trigger_id": mechanism.trigger_id,
                "gate_id": mechanism.gate_id,
                "active": mechanism.latched,
                "progress": max(
                    0.0,
                    min(1.0, float(self.data.qpos[mechanism.joint_qpos_address]) / mechanism.travel),
                ),
            }
            for mechanism in self.mechanisms
        ]

    def active_zone_ids(self, semantic_type: str, *, subject_id: str | None = None) -> list[str]:
        subject_id = subject_id or self.agent.object_id
        return [
            zone.id
            for candidate_subject_id, zone in self._current_zone_pairs()
            if candidate_subject_id == subject_id and zone.semantic_type == semantic_type
        ]

    def zone_events_since(self, sequence: int = 0) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self._event_log if event.sequence > sequence]

    def is_grounded(self) -> bool:
        return self._agent_contact_state().grounded

    def has_ground_support(self) -> bool:
        """Return stable player-facing grounded state across contact flicker."""

        return self.is_grounded() or (
            float(self.data.time) - self._last_grounded_at <= JUMP_COYOTE_SECONDS
        )

    def has_blocking_contact(self) -> bool:
        """Return whether the agent is touching a solid wall-like collider."""

        return bool(self._agent_contact_state().blocking_normals)

    def movement_is_blocked(
        self,
        *,
        right: float,
        forward: float,
        camera_azimuth: float,
    ) -> bool:
        """Return whether current solid contacts oppose the requested movement."""

        desired_velocity = camera_relative_velocity(
            camera_azimuth,
            right=right,
            forward=forward,
            speed=MOVE_SPEED,
        )
        if math.hypot(*desired_velocity) <= 1e-6:
            return False
        return any(
            desired_velocity[0] * normal_x + desired_velocity[1] * normal_y > 1e-6
            for normal_x, normal_y in self._agent_contact_state().blocking_normals
        )

    def _agent_contact_state(self) -> _AgentContactState:
        center_z = float(self.data.xpos[self.agent.body_id][2])
        highest_ground_contact = center_z - self.agent.height * 0.08
        lowest_ceiling_contact = center_z + self.agent.height * 0.08
        grounded = False
        ceiling_blocked = False
        blocking_normals: list[tuple[float, float]] = []
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            agent_is_geom1 = geom1 in self.agent.geom_ids
            agent_is_geom2 = geom2 in self.agent.geom_ids
            if not agent_is_geom1 and not agent_is_geom2:
                continue
            direction = 1.0 if agent_is_geom1 else -1.0
            normal = tuple(direction * float(value) for value in contact.frame[:3])
            normal_z = abs(normal[2])
            contact_z = float(contact.pos[2])
            if normal_z >= GROUND_NORMAL_Z_MIN and contact_z <= highest_ground_contact:
                grounded = True
            elif normal_z >= GROUND_NORMAL_Z_MIN and contact_z >= lowest_ceiling_contact:
                ceiling_blocked = True
            if normal_z >= GROUND_NORMAL_Z_MIN:
                continue
            other_geom = geom2 if agent_is_geom1 else geom1
            if other_geom not in self._blocking_geom_ids:
                continue
            horizontal_length = math.hypot(normal[0], normal[1])
            if horizontal_length <= 1e-8:
                continue
            blocking_normals.append((normal[0] / horizontal_length, normal[1] / horizontal_length))
        return _AgentContactState(
            grounded=grounded,
            blocking_normals=_deduplicate_normals(blocking_normals),
            ceiling_blocked=ceiling_blocked,
        )

    def body_transforms(self) -> list[dict[str, Any]]:
        return [
            {
                "id": object_id,
                "position": [float(value) for value in self.data.xpos[body_id]],
                "rotation_matrix": [float(value) for value in self.data.xmat[body_id]],
            }
            for object_id, body_id in self.dynamic_body_ids.items()
        ]

    def _update_mechanisms(self) -> None:
        for mechanism in self.mechanisms:
            if not mechanism.latched and self._switch_occupied(mechanism.trigger_id):
                mechanism.latched = True
            self.data.ctrl[mechanism.actuator_id] = mechanism.travel if mechanism.latched else 0.0

    def _record_zone_events(self, *, initial: bool = False) -> None:
        current_pairs = {
            (subject_id, zone.id)
            for subject_id, zone in self._current_zone_pairs()
        }
        entered = current_pairs if initial else current_pairs - self._active_zone_pairs
        exited = set() if initial else self._active_zone_pairs - current_pairs
        zones = {zone.id: zone for zone in self.sensor_zones}
        for event_type, pairs in (("zone_entered", entered), ("zone_exited", exited)):
            for subject_id, zone_id in sorted(pairs):
                zone = zones[zone_id]
                self._event_sequence += 1
                event = RuntimeZoneEvent(
                    sequence=self._event_sequence,
                    step=self.step_count,
                    simulation_time=float(self.data.time),
                    type=event_type,
                    subject_id=subject_id,
                    zone_id=zone_id,
                    semantic_type=zone.semantic_type,
                )
                self._event_log.append(event)
                if event_type == "zone_entered" and subject_id == self.agent.object_id:
                    if zone.semantic_type == "hazard":
                        self.hazard_entries += 1
                    elif zone.semantic_type == "goal":
                        self.goal_entries += 1
                        self.goal_reached = True
        self._active_zone_pairs = current_pairs

    def _current_zone_pairs(self) -> list[tuple[str, EnvObject3D]]:
        pairs: list[tuple[str, EnvObject3D]] = []
        for subject_id, body_id in self.dynamic_body_ids.items():
            subject = self.object_specs[subject_id]
            if subject.body_type != "dynamic":
                continue
            position = tuple(float(value) for value in self.data.xpos[body_id])
            yaw = yaw_from_rotation_matrix(
                self.data.xmat[body_id],
                fallback=float(subject.yaw),
            )
            for zone in self.sensor_zones:
                if _object_overlaps_zone(position, subject, zone, yaw=yaw):
                    pairs.append((subject_id, zone))
        return pairs

    def _switch_occupied(self, trigger_id: str) -> bool:
        zone = self.switch_zones[trigger_id]
        agent_yaw = yaw_from_rotation_matrix(
            self.data.xmat[self.agent.body_id],
            fallback=float(self.agent_spec.yaw),
        )
        if _object_overlaps_zone(
            self.agent_position(),
            self.agent_spec,
            zone,
            yaw=agent_yaw,
        ):
            return True
        for object_id, body_id in self.dynamic_body_ids.items():
            if object_id == self.agent.object_id or self.object_specs[object_id].body_type == "mechanism":
                continue
            position = tuple(float(value) for value in self.data.xpos[body_id])
            obj = self.object_specs[object_id]
            yaw = yaw_from_rotation_matrix(
                self.data.xmat[body_id],
                fallback=float(obj.yaw),
            )
            if _object_overlaps_zone(position, obj, zone, yaw=yaw):
                return True
        return False

    def _keep_agent_upright(self) -> None:
        qpos = self.agent.qpos_address
        half_yaw = self._heading_yaw / 2.0
        self.data.qpos[qpos + 3 : qpos + 7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def resolve_scene_paths(source: str | Path) -> tuple[Path, Path, Path]:
    path = Path(source).expanduser().resolve()
    scene_dir = path if path.is_dir() else path.parent
    xml_path = scene_dir / WORLD_XML_FILENAME if path.is_dir() else path
    spec_path = scene_dir / ENV_SPEC_FILENAME
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo world does not exist: {xml_path}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"environment spec does not exist: {spec_path}")
    return scene_dir, xml_path, spec_path


def camera_relative_velocity(
    camera_azimuth: float,
    *,
    right: float,
    forward: float,
    speed: float,
) -> tuple[float, float]:
    magnitude = math.hypot(right, forward)
    if magnitude > 1.0:
        right /= magnitude
        forward /= magnitude
    radians = math.radians(camera_azimuth)
    screen_right = (math.cos(radians), -math.sin(radians))
    screen_forward = (-math.sin(radians), -math.cos(radians))
    return (
        speed * (right * screen_right[0] + forward * screen_forward[0]),
        speed * (right * screen_right[1] + forward * screen_forward[1]),
    )


def _move_toward_velocity(
    current: tuple[float, float],
    target: tuple[float, float],
    *,
    max_delta: float,
) -> tuple[float, float]:
    delta_x = target[0] - current[0]
    delta_y = target[1] - current[1]
    distance = math.hypot(delta_x, delta_y)
    if distance <= max_delta or distance <= 1e-9:
        return target
    scale = max_delta / distance
    return current[0] + delta_x * scale, current[1] + delta_y * scale


def _project_out_of_contacts(
    velocity: tuple[float, float],
    blocking_normals: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    velocity_x, velocity_y = velocity
    # Repeating the projections resolves corners where satisfying one contact
    # can introduce a small inward component along another contact normal.
    for _ in range(max(1, len(blocking_normals))):
        changed = False
        for normal_x, normal_y in blocking_normals:
            inward_speed = velocity_x * normal_x + velocity_y * normal_y
            if inward_speed <= 0.0:
                continue
            velocity_x -= inward_speed * normal_x
            velocity_y -= inward_speed * normal_y
            changed = True
        if not changed:
            break
    return velocity_x, velocity_y


def _deduplicate_normals(normals: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique: list[tuple[float, float]] = []
    for normal in normals:
        if any(normal[0] * other[0] + normal[1] * other[1] >= 0.995 for other in unique):
            continue
        unique.append(normal)
    return tuple(unique)


def run_window(simulation: PlayableSimulation) -> None:
    try:
        import glfw
        import mujoco
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError(f"MuJoCo viewer dependencies are unavailable: {exc}") from exc

    active_until: dict[int, float] = {}
    reset_requested = threading.Event()
    movement_keys = {
        glfw.KEY_W: (0.0, 1.0),
        glfw.KEY_UP: (0.0, 1.0),
        glfw.KEY_S: (0.0, -1.0),
        glfw.KEY_DOWN: (0.0, -1.0),
        glfw.KEY_D: (1.0, 0.0),
        glfw.KEY_RIGHT: (1.0, 0.0),
        glfw.KEY_A: (-1.0, 0.0),
        glfw.KEY_LEFT: (-1.0, 0.0),
    }
    jump_until = 0.0

    def on_key(key: int) -> None:
        nonlocal jump_until
        if key == glfw.KEY_R:
            reset_requested.set()
        elif key == glfw.KEY_SPACE:
            jump_until = time.monotonic() + INPUT_GRACE_SECONDS
        elif key in movement_keys:
            active_until[key] = time.monotonic() + INPUT_GRACE_SECONDS

    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = simulation.agent_position()
        viewer.cam.distance = max(7.0, float(simulation.model.stat.extent) * 0.75)
        viewer.cam.azimuth = -90.0
        viewer.cam.elevation = -32.0
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "PLAY MODE",
                "Tap/hold WASD or arrows to move\nSpace  Jump\nR  Reset\nMouse  Orbit / Pan / Zoom\nEsc  Close",
            )
        )
        previous = time.monotonic()
        accumulator = 0.0
        last_status = ""
        while viewer.is_running():
            now = time.monotonic()
            accumulator = min(accumulator + now - previous, 0.1)
            previous = now
            if reset_requested.is_set():
                simulation.reset()
                reset_requested.clear()
                active_until.clear()
                jump_until = 0.0

            right_input = 0.0
            forward_input = 0.0
            for key, deadline in list(active_until.items()):
                if deadline < now:
                    del active_until[key]
                    continue
                right_delta, forward_delta = movement_keys[key]
                right_input += right_delta
                forward_input += forward_delta
            while accumulator >= simulation.model.opt.timestep:
                simulation.step(
                    right=right_input,
                    forward=forward_input,
                    camera_azimuth=float(viewer.cam.azimuth),
                    jump=jump_until >= now,
                )
                accumulator -= simulation.model.opt.timestep
            status = simulation.status()
            if status != last_status:
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "PLAY MODE",
                            "Tap/hold WASD or arrows to move\nSpace  Jump\nR  Reset\nMouse  Orbit / Pan / Zoom\nEsc  Close",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                            "STATUS",
                            status,
                        ),
                    ]
                )
                last_status = status
            viewer.sync()
            time.sleep(0.005)


def smoke_test(source: str | Path, *, steps: int = 30) -> dict[str, Any]:
    simulation = PlayableSimulation.from_scene(source)
    start = simulation.agent_position()
    for _ in range(max(1, steps)):
        simulation.step(right=0.0, forward=1.0, camera_azimuth=-90.0)
    end = simulation.agent_position()
    return {
        "env_id": simulation.spec.id,
        "agent_id": simulation.agent.object_id,
        "start": list(start),
        "end": list(end),
        "moved": math.dist(start[:2], end[:2]) > 0.01,
    }


def _find_agent_control(model: Any, spec: EnvSpec3D, mujoco_module: Any) -> AgentControl:
    agent = next((obj for obj in spec.objects if obj.semantic_type == "agent"), None)
    if agent is None:
        raise ValueError("the environment has no playable agent")
    joint_name = f"{agent.id}_freejoint"
    joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"playable agent joint {joint_name!r} was not found in world.xml")
    body_id = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, agent.id))
    if body_id < 0:
        raise ValueError(f"playable agent body {agent.id!r} was not found in world.xml")
    geom_ids = frozenset(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) == body_id
    )
    if not geom_ids:
        raise ValueError(f"playable agent body {agent.id!r} does not contain a collision geom")
    return AgentControl(
        object_id=agent.id,
        body_id=body_id,
        geom_ids=geom_ids,
        qpos_address=int(model.jnt_qposadr[joint_id]),
        qvel_address=int(model.jnt_dofadr[joint_id]),
        height=float(agent.size[2]),
    )


def _find_dynamic_body_ids(model: Any, spec: EnvSpec3D, mujoco_module: Any) -> dict[str, int]:
    body_ids: dict[str, int] = {}
    for obj in spec.objects:
        if obj.body_type not in {"dynamic", "mechanism"}:
            continue
        body_id = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, obj.id))
        if body_id < 0:
            raise ValueError(f"dynamic body {obj.id!r} was not found in world.xml")
        body_ids[obj.id] = body_id
    return body_ids


def _find_blocking_geom_ids(model: Any, spec: EnvSpec3D, mujoco_module: Any) -> frozenset[int]:
    geom_ids: set[int] = set()
    for obj in spec.objects:
        if obj.body_type not in {"static", "mechanism"}:
            continue
        geom_id = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_GEOM, obj.id))
        if geom_id < 0:
            raise ValueError(f"blocking geom {obj.id!r} was not found in world.xml")
        geom_ids.add(geom_id)
    return frozenset(geom_ids)


def _find_mechanism_controls(model: Any, spec: EnvSpec3D, mujoco_module: Any) -> list[MechanismControl]:
    objects = {obj.id: obj for obj in spec.objects}
    controls: list[MechanismControl] = []
    for mechanism in spec.mechanisms:
        gate = objects[mechanism.gate_id]
        actuator_id = int(
            mujoco_module.mj_name2id(
                model,
                mujoco_module.mjtObj.mjOBJ_ACTUATOR,
                f"{gate.id}_actuator",
            )
        )
        joint_id = int(
            mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, f"{gate.id}_slide")
        )
        if actuator_id < 0 or joint_id < 0:
            raise ValueError(f"mechanism {mechanism.id!r} is missing its compiled gate controls")
        controls.append(
            MechanismControl(
                id=mechanism.id,
                trigger_id=mechanism.trigger_id,
                gate_id=mechanism.gate_id,
                actuator_id=actuator_id,
                joint_qpos_address=int(model.jnt_qposadr[joint_id]),
                travel=float(gate.metadata.get("travel") or float(gate.size[2]) + 0.2),
            )
        )
    return controls


def _object_overlaps_zone(
    position: tuple[float, float, float],
    obj: EnvObject3D,
    zone: EnvObject3D,
    *,
    yaw: float,
) -> bool:
    runtime_object = _RuntimeShape(
        shape=str(obj.shape),
        position=tuple(float(value) for value in position),
        size=tuple(float(value) for value in obj.size),
        yaw=float(yaw),
    )
    runtime_zone = _RuntimeShape(
        shape=str(zone.shape),
        position=tuple(float(value) for value in zone.position),
        size=tuple(float(value) for value in zone.size),
        yaw=float(zone.yaw),
    )
    return volumes_overlap(runtime_object, runtime_zone)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a generated Environment Generation MuJoCo environment.")
    parser.add_argument("scene", type=Path, help="Environment directory or world.xml path")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short headless movement test and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        print(json.dumps(smoke_test(args.scene), separators=(",", ":")))
        return
    run_window(PlayableSimulation.from_scene(args.scene))


if __name__ == "__main__":
    main()
