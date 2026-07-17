from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .episode_runtime import decide_episode
from .player import PlayableSimulation


PLAY_SESSION_TTL_SECONDS = 10 * 60
MAX_ADVANCE_SECONDS = 0.1


@dataclass
class PlaySession:
    id: str
    env_id: str
    simulation: PlayableSimulation
    clock: Callable[[], float]
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_step_at: float = field(init=False)
    last_access_at: float = field(init=False)
    terminal_state: str = field(default="", init=False)
    terminal_reason: str = field(default="", init=False)
    reset_count: int = field(default=0, init=False)
    event_cursor: int = field(default=0, init=False)
    pending_events: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self.last_step_at = now
        self.last_access_at = now
        self._queue_zone_events()

    def advance(self, *, right: float, forward: float, camera_azimuth: float, jump: bool) -> dict[str, Any]:
        with self.lock:
            now = self.clock()
            if self.terminal_state:
                self.last_access_at = now
                return self.snapshot()
            elapsed = max(float(self.simulation.model.opt.timestep), min(MAX_ADVANCE_SECONDS, now - self.last_step_at))
            steps = max(1, int(round(elapsed / float(self.simulation.model.opt.timestep))))
            for _ in range(steps):
                self.simulation.step(
                    right=right,
                    forward=forward,
                    camera_azimuth=camera_azimuth,
                    jump=jump,
                )
                decision = decide_episode(
                    safety_failure=self.simulation.safety_failure_reason(),
                )
                if decision.terminal:
                    self.terminal_state = decision.outcome
                    self.terminal_reason = decision.reason
                    break
            self._queue_zone_events()
            self.last_step_at = now
            self.last_access_at = now
            return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.simulation.reset()
            self.terminal_state = ""
            self.terminal_reason = ""
            self.reset_count += 1
            self.event_cursor = 0
            self.pending_events = []
            self._queue_zone_events()
            now = self.clock()
            self.last_step_at = now
            self.last_access_at = now
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        self._queue_zone_events()
        events = self.pending_events
        self.pending_events = []
        game = self.simulation.game_state(events=events)
        if self.terminal_state:
            game["state"] = self.terminal_state
            game["failure_reason"] = self.terminal_reason
        game["metrics"] = {**game["metrics"], "resets": self.reset_count}
        return {
            "session_id": self.id,
            "env_id": self.env_id,
            "simulation_time": float(self.simulation.data.time),
            "status": self.simulation.status(),
            "grounded": self.simulation.is_grounded(),
            "agent_id": self.simulation.agent.object_id,
            "objects": self.simulation.body_transforms(),
            "game": game,
        }

    def _queue_zone_events(self) -> None:
        events = self.simulation.zone_events_since(self.event_cursor)
        if events:
            self.pending_events.extend(events)
            self.event_cursor = int(events[-1]["sequence"])


class PlaySessionManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = PLAY_SESSION_TTL_SECONDS,
    ) -> None:
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, PlaySession] = {}
        self._lock = threading.Lock()

    def start(self, *, scene_dir: Path) -> dict[str, Any]:
        simulation = PlayableSimulation.from_scene(scene_dir)
        session_id = uuid.uuid4().hex
        session = PlaySession(
            id=session_id,
            env_id=simulation.spec.id,
            simulation=simulation,
            clock=self.clock,
        )
        with self._lock:
            self._remove_expired_locked()
            self._sessions[session_id] = session
        return session.snapshot()

    def step(
        self,
        session_id: str,
        *,
        right: Any = 0.0,
        forward: Any = 0.0,
        camera_azimuth: Any = 0.0,
        jump: Any = False,
    ) -> dict[str, Any]:
        session = self._get(session_id)
        return session.advance(
            right=_bounded_number(right, minimum=-1.0, maximum=1.0, field_name="right"),
            forward=_bounded_number(forward, minimum=-1.0, maximum=1.0, field_name="forward"),
            camera_azimuth=_bounded_number(
                camera_azimuth,
                minimum=-10000.0,
                maximum=10000.0,
                field_name="camera_azimuth",
            ),
            jump=_boolean(jump, field_name="jump"),
        )

    def reset(self, session_id: str) -> dict[str, Any]:
        return self._get(session_id).reset()

    def stop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def has_env(self, env_id: str) -> bool:
        with self._lock:
            self._remove_expired_locked()
            return any(session.env_id == env_id for session in self._sessions.values())

    def _get(self, session_id: str) -> PlaySession:
        with self._lock:
            self._remove_expired_locked()
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("play session was not found or has expired")
        return session

    def _remove_expired_locked(self) -> None:
        cutoff = self.clock() - self.ttl_seconds
        expired = [session_id for session_id, session in self._sessions.items() if session.last_access_at < cutoff]
        for session_id in expired:
            del self._sessions[session_id]


def _bounded_number(value: Any, *, minimum: float, maximum: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return max(minimum, min(maximum, number))


def _boolean(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean")
