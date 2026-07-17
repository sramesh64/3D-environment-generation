from __future__ import annotations

import math

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_verification import scene_objects
from environment_generation.player import (
    JUMP_SPEED,
    MOVE_SPEED,
    PlayableSimulation,
    camera_relative_velocity,
    smoke_test,
)
from environment_generation.trajectory_assertions import capture_trajectory_frame, evaluate_assertion_group


def _persist_playable_scene(tmp_path):
    builder = EnvSpec3DBuilder("playable", description="playable box and goal scene")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "playable"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir


def _persist_ramp_scene(tmp_path):
    builder = EnvSpec3DBuilder("ramp_playable", description="ground-to-platform ramp")
    builder.add_ground_plane(14, 8, id="ground")
    builder.add_platform(2.5, 0, z=0, width=3, depth=3, thickness=1, id="platform")
    builder.add_ramp(-2, 0, length=3, width=2, rise=1, thickness=0.25, id="ramp")
    builder.add_agent_spawn(-3.2, 0, id="agent")
    scene_dir = tmp_path / "ramp_playable"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir


def _persist_blocker_scene(tmp_path, blocker_kind: str):
    builder = EnvSpec3DBuilder(f"{blocker_kind}_contact", description="vertical contact regression scene")
    builder.add_ground_plane(12, 8, id="ground")
    builder.add_agent_spawn(0, 0, id="agent")
    if blocker_kind == "wall":
        builder.add_wall(1, 0, width=0.3, depth=6, height=2, id="blocker")
    elif blocker_kind == "static_box":
        builder.add_static_box(1, 0, width=0.3, depth=6, height=2, id="blocker")
    elif blocker_kind == "platform":
        builder.add_platform(1, 0, z=0, width=0.3, depth=6, thickness=2, id="blocker")
    elif blocker_kind == "gate":
        builder.add_sliding_gate(1, 0, width=0.3, depth=6, height=2, id="blocker")
    else:  # pragma: no cover - helper misuse
        raise ValueError(f"unsupported blocker kind {blocker_kind!r}")
    scene_dir = tmp_path / f"{blocker_kind}_contact"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir


def test_camera_relative_velocity_tracks_visible_camera_axes() -> None:
    forward = camera_relative_velocity(-90.0, right=0.0, forward=1.0, speed=MOVE_SPEED)
    right = camera_relative_velocity(-90.0, right=1.0, forward=0.0, speed=MOVE_SPEED)
    diagonal = camera_relative_velocity(-90.0, right=1.0, forward=1.0, speed=MOVE_SPEED)

    assert forward == pytest.approx((MOVE_SPEED, 0.0), abs=1e-9)
    assert right == pytest.approx((0.0, MOVE_SPEED), abs=1e-9)
    assert math.hypot(*diagonal) == pytest.approx(MOVE_SPEED)


def test_camera_relative_forward_is_perpendicular_to_screen_right() -> None:
    azimuth = -42.0
    forward = camera_relative_velocity(azimuth, right=0.0, forward=1.0, speed=1.0)
    right = camera_relative_velocity(azimuth, right=1.0, forward=0.0, speed=1.0)

    assert forward[0] * right[0] + forward[1] * right[1] == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(*forward) == pytest.approx(1.0)
    assert math.hypot(*right) == pytest.approx(1.0)


def test_playable_simulation_moves_agent_through_mujoco(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir = _persist_playable_scene(tmp_path)
    simulation = PlayableSimulation.from_scene(scene_dir)
    start = simulation.agent_position()

    for _ in range(40):
        simulation.step(right=0.0, forward=1.0, camera_azimuth=-90.0)

    end = simulation.agent_position()
    assert math.dist(start[:2], end[:2]) > 0.05
    assert end[0] > start[0]
    assert end[2] == pytest.approx(start[2], abs=0.2)


def test_player_smoke_test_reports_agent_movement(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir = _persist_playable_scene(tmp_path)

    result = smoke_test(scene_dir, steps=40)

    assert result["env_id"] == "playable"
    assert result["agent_id"] == "agent"
    assert result["moved"] is True


def test_runtime_goal_events_match_typed_shape_aware_overlap(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("corner_goal", description="shape-aware goal boundary")
    builder.set_world(theme="robot_courtyard")
    builder.add_ground_plane(24, 18, id="ground")
    agent_id = builder.add_agent_spawn(-11, 0, id="agent")
    goal_id = builder.add_goal_zone(0, 0, id="goal")
    builder.configure_reach_goal_game(agent_id, goal_id)
    scene_dir = tmp_path / "corner_goal"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    simulation = PlayableSimulation.from_scene(scene_dir)
    objects = scene_objects(simulation.spec)
    objective = {
        "mode": "all",
        "conditions": [
            {
                "id": "enter_goal",
                "description": "The round agent overlaps the box goal.",
                "temporal": "eventually",
                "predicate": {
                    "type": "overlap",
                    "subject": {"id": "agent"},
                    "target": {"id": "goal"},
                },
            }
        ],
    }
    address = simulation.agent.qpos_address

    simulation.data.qpos[address : address + 3] = [1.04, 1.04, 0.55]
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    outside_corner = capture_trajectory_frame(simulation, reset_count=0)
    outside_result = evaluate_assertion_group(
        group=objective,
        frames=[outside_corner],
        objects=objects,
    )

    assert simulation.active_zone_ids("goal") == []
    assert "goal" not in outside_corner["terminal_events"]
    assert outside_result["satisfied"] is False

    simulation.data.qpos[address : address + 3] = [0.9, 0.9, 0.55]
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    inside_corner = capture_trajectory_frame(simulation, reset_count=0)
    inside_result = evaluate_assertion_group(
        group=objective,
        frames=[inside_corner],
        objects=objects,
    )

    assert simulation.active_zone_ids("goal") == ["goal"]
    assert "goal" in inside_corner["terminal_events"]
    assert inside_result["satisfied"] is True


def test_jump_is_grounded_rises_and_lands_without_air_retrigger(tmp_path) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_playable_scene(tmp_path))
    simulation.step(right=0, forward=0, camera_azimuth=-42, jump=False)
    start_z = simulation.agent_position()[2]

    assert simulation.is_grounded() is True
    heights = []
    for _ in range(180):
        simulation.step(right=0, forward=0, camera_azimuth=-42, jump=True)
        heights.append(simulation.agent_position()[2])

    assert max(heights) > start_z + 0.8
    assert max(heights) < start_z + 1.3
    assert simulation.is_grounded() is True
    assert simulation.agent_position()[2] == pytest.approx(start_z, abs=0.03)


def test_jump_can_trigger_again_after_release_and_landing(tmp_path) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_playable_scene(tmp_path))
    simulation.step(right=0, forward=0, camera_azimuth=-42, jump=False)
    simulation.step(right=0, forward=0, camera_azimuth=-42, jump=True)
    simulation.step(right=0, forward=0, camera_azimuth=-42, jump=False)

    for _ in range(180):
        simulation.step(right=0, forward=0, camera_azimuth=-42, jump=False)
        if simulation.is_grounded():
            break

    simulation.step(right=0, forward=0, camera_azimuth=-42, jump=True)

    vertical_velocity = float(simulation.data.qvel[simulation.agent.qvel_address + 2])
    assert vertical_velocity > 4.0


def test_agent_traverses_and_jumps_on_ramp_without_contact_launch(tmp_path) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_ramp_scene(tmp_path))
    heights: list[float] = []
    vertical_velocities: list[float] = []
    jumped = False

    for _ in range(260):
        jump = not jumped and simulation.agent_position()[0] > -1.2
        simulation.step(right=0, forward=1, camera_azimuth=-90, jump=jump)
        jumped = jumped or jump
        heights.append(simulation.agent_position()[2])
        vertical_velocities.append(float(simulation.data.qvel[simulation.agent.qvel_address + 2]))

    assert jumped is True
    assert simulation.jump_count == 1
    assert max(heights) > 1.8
    assert max(heights) < 2.3
    assert max(vertical_velocities) <= JUMP_SPEED + 0.1


@pytest.mark.parametrize("blocker_kind", ["wall", "static_box", "platform", "gate"])
def test_jump_remains_ballistic_while_pressing_into_vertical_blocker(tmp_path, blocker_kind) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_blocker_scene(tmp_path, blocker_kind))

    for _ in range(80):
        simulation.step(right=1, forward=0, camera_azimuth=0, jump=False)

    start_z = simulation.agent_position()[2]
    assert simulation.is_grounded() is True

    heights: list[float] = []
    for step in range(180):
        simulation.step(right=1, forward=0, camera_azimuth=0, jump=step == 0)
        heights.append(simulation.agent_position()[2])

    assert simulation.jump_count == 1
    assert max(heights) > start_z + 0.8
    assert max(heights) < start_z + 1.3
    assert simulation.is_grounded() is True
    assert simulation.agent_position()[2] == pytest.approx(start_z, abs=0.03)


def test_agent_moves_parallel_to_wall_without_pressing_through_it(tmp_path) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_blocker_scene(tmp_path, "wall"))
    start = simulation.agent_position()

    for _ in range(140):
        simulation.step(right=1, forward=-1, camera_azimuth=0, jump=False)

    end = simulation.agent_position()
    assert end[0] < 0.65
    assert end[1] > start[1] + 1.5
    assert end[2] == pytest.approx(start[2], abs=0.03)
    assert simulation.is_grounded() is True


def test_blocking_contact_is_directional_for_requested_movement(tmp_path) -> None:
    pytest.importorskip("mujoco")
    simulation = PlayableSimulation.from_scene(_persist_blocker_scene(tmp_path, "wall"))

    for _ in range(80):
        simulation.step(right=1, forward=0, camera_azimuth=0, jump=False)

    assert simulation.has_blocking_contact() is True
    assert simulation.movement_is_blocked(right=1, forward=0, camera_azimuth=0) is True
    assert simulation.movement_is_blocked(right=-1, forward=0, camera_azimuth=0) is False
    assert simulation.movement_is_blocked(right=0, forward=1, camera_azimuth=0) is False


def test_contact_aware_movement_still_pushes_dynamic_objects(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("dynamic_push", description="push through a dynamic contact")
    builder.add_ground_plane(12, 8, id="ground")
    builder.add_agent_spawn(-1.2, 0, id="agent")
    builder.add_pushable_box(0, 0, width=0.9, depth=0.9, height=0.9, id="box")
    scene_dir = tmp_path / "dynamic_push"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    simulation = PlayableSimulation.from_scene(scene_dir)
    box_body_id = simulation.dynamic_body_ids["box"]
    start_box_x = float(simulation.data.xpos[box_body_id][0])
    blocking_reports: list[bool] = []

    for _ in range(160):
        simulation.step(right=1, forward=0, camera_azimuth=0, jump=False)
        blocking_reports.append(
            simulation.has_blocking_contact()
            or simulation.movement_is_blocked(right=1, forward=0, camera_azimuth=0)
        )

    assert float(simulation.data.xpos[box_body_id][0]) > start_box_x + 0.8
    assert not any(blocking_reports)
    assert simulation.is_grounded() is True


def test_ballistic_jump_still_stops_at_an_overhead_surface(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("low_ceiling", description="jump below a low platform")
    builder.add_ground_plane(10, 8, id="ground")
    builder.add_agent_spawn(0, 0, id="agent")
    builder.add_platform(0, 0, z=1.45, width=4, depth=4, thickness=0.25, id="ceiling")
    scene_dir = tmp_path / "low_ceiling"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    simulation = PlayableSimulation.from_scene(scene_dir)
    heights: list[float] = []

    for step in range(180):
        simulation.step(right=0, forward=0, camera_azimuth=0, jump=step == 0)
        heights.append(simulation.agent_position()[2])

    assert simulation.jump_count == 1
    assert max(heights) < 0.93
    assert simulation.is_grounded() is True
