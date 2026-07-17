from __future__ import annotations

import math

import pytest

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_verification import (
    EnvVerificationError,
    env_verification_summary,
    finalization_block,
    normalize_env_verification_plan,
    run_env_verification,
    spec_hash,
    write_env_verification_plan,
    write_env_verification_report,
)


def _simple_spec(*, floating_box: bool = False) -> dict:
    builder = EnvSpec3DBuilder("verified_env", description="verification scene")
    builder.add_ground_plane(12, 8, id="ground")
    builder.add_agent_spawn(-3, 0, id="agent")
    builder.add_goal_zone(4, 0, id="goal")
    builder.add_pushable_box(0, 0, z=2 if floating_box else 0, id="box")
    builder.add_cylinder(0, 2.5, id="front_cylinder")
    builder.add_platform(0, -2.5, z=1.0, id="platform")
    return builder.finalize().model_dump(mode="json")


def _plan(checks: list[dict], spec: dict | None = None) -> dict:
    return normalize_env_verification_plan(
        env_id="verified_env",
        prompt="make a 3D scene",
        checks=checks,
        operation_count=4,
        draft_spec=spec or _simple_spec(),
    )


def _screen_context() -> dict:
    return {
        "review_id": "turn-screen-test",
        "prompt": "add an agent in the bottom left",
        "view_context": {
            "screen_space": {
                "camera": {
                    "position": [3.432, 11.789, 9.46],
                    "target": [0.0, 0.0, 0.5],
                    "fov_y_degrees": 45.0,
                    "aspect": 976 / 547,
                },
                "regions": {
                    "bottom_left": {
                        "bounds_uv": {"left": 0.05, "top": 0.58, "right": 0.34, "bottom": 0.88},
                        "anchor": {"world_position": [-5.0, 4.8, 0.0]},
                    }
                },
            }
        },
    }


def test_verification_plan_rejects_unsupported_check_shapes() -> None:
    with pytest.raises(EnvVerificationError, match="unsupported type"):
        _plan([{"type": "unsupported"}])

    with pytest.raises(EnvVerificationError, match="selector"):
        _plan([{"type": "object_count", "selector": {"selector": "agent"}}])

    with pytest.raises(EnvVerificationError, match="got keys: type"):
        _plan([{"type": "object_count", "selector": {"type": "agent"}}])


def test_verification_plan_accepts_selector_shorthand_and_aliases() -> None:
    plan = _plan(
        [
            {"type": "object_count", "selector": "agent"},
            {"type": "object_count", "selector": {"object_type": "goal"}},
            {"type": "object_count", "selector": {"body_type": "dynamic", "shape": "box"}},
        ]
    )

    assert plan["checks"][0]["selector"] == {"semantic_type": "agent"}
    assert plan["checks"][1]["selector"] == {"semantic_type": "goal"}
    assert plan["checks"][2]["selector"] == {"body_type": "dynamic", "shape": "box"}


def test_screen_region_uses_frozen_submitted_camera_instead_of_world_axes() -> None:
    correct = _simple_spec()
    agent = next(obj for obj in correct["objects"] if obj["id"] == "agent")
    agent["position"] = [-5.0, 4.5, 0.55]
    plan = normalize_env_verification_plan(
        env_id="verified_env",
        prompt="add an agent in the bottom left",
        checks=[
            {
                "id": "agent_in_submitted_bottom_left",
                "type": "screen_region",
                "subject": {"id": "agent"},
                "region": "bottom_left",
            }
        ],
        operation_count=4,
        draft_spec=correct,
        screen_context=_screen_context(),
    )

    passed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=correct,
        final_spec=correct,
        operation_count=4,
    )
    assert passed["status"] == "passed"
    assert passed["results"][0]["metrics"]["subjects"][0]["screen_uv"][1] > 0.58

    wrong = _simple_spec()
    wrong_agent = next(obj for obj in wrong["objects"] if obj["id"] == "agent")
    wrong_agent["position"] = [-5.0, -1.2, 0.55]
    failed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=wrong,
        final_spec=wrong,
        operation_count=4,
    )
    assert failed["status"] == "failed"
    assert failed["results"][0]["metrics"]["subjects"][0]["screen_uv"][1] < 0.58
    assert "world-space anchor" in failed["results"][0]["repair_hints"][0]


def test_screen_region_requires_submitted_camera_context() -> None:
    with pytest.raises(EnvVerificationError, match="exact submitted Studio camera"):
        _plan([{"type": "screen_region", "subject": "agent", "region": "bottom_left"}])


def test_screen_relation_projects_both_objects_through_submitted_camera() -> None:
    spec = _simple_spec()
    agent = next(obj for obj in spec["objects"] if obj["id"] == "agent")
    agent["position"] = [-5.0, 4.5, 0.55]
    plan = normalize_env_verification_plan(
        env_id="verified_env",
        prompt="put the agent left of the box",
        checks=[
            {
                "type": "screen_relation",
                "subject": {"id": "agent"},
                "target": {"id": "box"},
                "relation": "left_of",
            }
        ],
        operation_count=4,
        draft_spec=spec,
        screen_context=_screen_context(),
    )
    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )

    assert report["status"] == "passed"
    pair = report["results"][0]["metrics"]["pairs"][0]
    assert pair["subject_uv"][0] < pair["target_uv"][0]


def test_object_count_reports_critical_and_advisory_failures() -> None:
    spec = _simple_spec()
    plan = _plan(
        [
            {"id": "one_agent", "type": "object_count", "selector": "agent", "exact": 1},
            {
                "id": "too_many_goals",
                "type": "object_count",
                "selector": "goal",
                "exact": 2,
                "severity": "advisory",
            },
        ],
        spec,
    )

    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )

    assert report["status"] == "advisory_issues"
    assert report["summary"]["critical_failures"] == 0
    assert report["summary"]["advisory_failures"] == 1
    assert report["results"][0]["passed"] is True
    assert report["results"][1]["metrics"]["count"] == 1


def test_spatial_relation_checks_3d_axes_and_distances() -> None:
    spec = _simple_spec()
    plan = _plan(
        [
            {"id": "agent_left", "type": "spatial_relation", "subject": "agent", "relation": "left_of", "target": "goal"},
            {
                "id": "cylinder_front",
                "type": "spatial_relation",
                "subject": {"id": "front_cylinder"},
                "relation": "in_front_of",
                "target": {"id": "box"},
            },
            {"id": "platform_above", "type": "spatial_relation", "subject": "platform", "relation": "above", "target": "ground"},
            {"id": "box_near_agent", "type": "spatial_relation", "subject": {"id": "box"}, "relation": "near", "target": "agent", "distance": 4.0},
            {"id": "box_between", "type": "spatial_relation", "subject": {"id": "box"}, "relation": "between", "targets": ["agent", "goal"]},
        ],
        spec,
    )

    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )

    assert report["status"] == "passed"
    assert all(result["passed"] for result in report["results"])

    failed_plan = _plan(
        [{"id": "goal_left", "type": "spatial_relation", "subject": "goal", "relation": "left_of", "target": "agent"}],
        spec,
    )
    failed = run_env_verification(
        env_id="verified_env",
        plan=failed_plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )
    assert failed["status"] == "failed"
    assert failed["results"][0]["passed"] is False


def test_inside_relation_uses_oriented_volume_instead_of_rotated_aabb() -> None:
    builder = EnvSpec3DBuilder("verified_env", description="oriented containment")
    builder.add_ground_plane(12, 12, id="ground")
    builder.add_static_box(
        0,
        0,
        width=8,
        depth=0.4,
        height=2,
        yaw=math.pi / 4,
        id="container",
    )
    builder.add_static_box(
        2.5,
        -2.5,
        width=0.2,
        depth=0.2,
        height=0.6,
        id="subject",
    )
    outside = builder.finalize().model_dump(mode="json")
    plan = _plan(
        [
            {
                "id": "contained",
                "type": "spatial_relation",
                "subject": {"id": "subject"},
                "relation": "inside",
                "target": {"id": "container"},
            }
        ],
        outside,
    )

    failed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=outside,
        final_spec=outside,
        operation_count=4,
    )
    assert failed["status"] == "failed"

    inside = builder.to_spec_dict()
    subject = next(obj for obj in inside["objects"] if obj["id"] == "subject")
    subject["position"] = [0, 0, 0.3]
    passed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=inside,
        final_spec=inside,
        operation_count=4,
    )
    assert passed["status"] == "passed"


def test_on_surface_and_support_contact_catch_floating_objects() -> None:
    spec = _simple_spec()
    plan = _plan(
        [
            {"id": "box_on_ground", "type": "spatial_relation", "subject": {"id": "box"}, "relation": "on_surface", "target": "ground"},
            {"id": "dynamics_supported", "type": "support_contact", "selector": {"body_type": "dynamic", "tag": "pushable"}},
        ],
        spec,
    )

    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )
    assert report["status"] == "passed"

    floating = _simple_spec(floating_box=True)
    failed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=floating,
        final_spec=floating,
        operation_count=4,
    )
    assert failed["status"] == "failed"
    assert failed["results"][1]["metrics"]["failures"][0]["issue"] == "floating"


def test_ramp_connection_checks_both_walkable_endpoints() -> None:
    builder = EnvSpec3DBuilder("verified_env", description="connected ramp")
    builder.add_ground_plane(12, 8, id="ground")
    builder.add_platform(4.5, 0, z=0, width=3, depth=3, thickness=1, id="platform")
    builder.add_ramp(0, 0, length=3, width=2, rise=1, thickness=0.25, id="ramp")
    connected = builder.finalize().model_dump(mode="json")
    plan = _plan(
        [
            {
                "id": "ramp_joins_platform",
                "type": "ramp_connection",
                "ramp": {"id": "ramp"},
                "low_surface": {"id": "ground"},
                "high_surface": {"id": "platform"},
            }
        ],
        connected,
    )

    passed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=connected,
        final_spec=connected,
        operation_count=4,
    )

    assert passed["status"] == "passed"
    measurement = passed["results"][0]["metrics"]["ramps"][0]
    assert measurement["low_end"] == pytest.approx([0, 0, 0])
    assert measurement["high_end"] == pytest.approx([3, 0, 1])
    assert measurement["high_surface"]["horizontal_gap"] == pytest.approx(0)
    assert measurement["high_surface"]["vertical_gap"] == pytest.approx(0)

    disconnected_builder = EnvSpec3DBuilder("verified_env", description="disconnected ramp")
    disconnected_builder.add_ground_plane(12, 8, id="ground")
    disconnected_builder.add_platform(5, 0, z=0, width=3, depth=3, thickness=1, id="platform")
    disconnected_builder.add_ramp(0, 0, length=3, width=2, rise=0.6, thickness=0.25, id="ramp")
    disconnected = disconnected_builder.finalize().model_dump(mode="json")
    failed = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=disconnected,
        final_spec=disconnected,
        operation_count=4,
    )

    assert failed["status"] == "failed"
    failed_measurement = failed["results"][0]["metrics"]["ramps"][0]
    assert failed_measurement["high_surface"]["horizontal_gap"] == pytest.approx(0.5)
    assert failed_measurement["high_surface"]["vertical_gap"] == pytest.approx(0.4)


def test_physics_probe_pushable_moves_in_mujoco() -> None:
    spec = _simple_spec()
    plan = _plan(
        [{"id": "box_moves", "type": "physics_probe", "probe": "pushable_moves", "object_id": "box"}],
        spec,
    )

    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )

    assert report["status"] == "passed"
    assert report["results"][0]["metrics"]["displacement_xy"] > 0.15


def test_stale_report_detection_uses_draft_hash(tmp_path) -> None:
    spec = _simple_spec()
    plan = _plan([{"type": "object_count", "selector": "agent", "exact": 1}], spec)
    report = run_env_verification(
        env_id="verified_env",
        plan=plan,
        draft_spec=spec,
        final_spec=spec,
        operation_count=4,
    )
    write_env_verification_plan(tmp_path, plan)
    write_env_verification_report(tmp_path, report)
    assert env_verification_summary(tmp_path, draft_hash=spec_hash(spec), operation_count=4)["status"] == "passed"

    changed = dict(spec)
    changed["description"] = "changed"
    assert env_verification_summary(tmp_path, draft_hash=spec_hash(changed), operation_count=4)["status"] == "stale"
    block = finalization_block(scene_dir=tmp_path, draft_hash=spec_hash(changed), operation_count=4)
    assert block is not None
    assert block["reason"] == "stale_report"
