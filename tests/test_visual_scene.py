from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from environment_generation.artifacts import (
    HISTORY_FILENAME,
    load_scene,
    persist_artifacts,
    refresh_visual_scene_artifact,
)
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_verification import (
    normalize_env_verification_plan,
    run_env_verification,
    write_env_verification_plan,
    write_env_verification_report,
)
from environment_generation.mujoco_compile import compile_spec_to_mjcf
from environment_generation.visual_scene import DEFAULT_VISUAL_THEME, VISUAL_SCENE_FILENAME, compile_visual_scene


def _coverage_spec():
    builder = EnvSpec3DBuilder("visual_scene", description="visual coverage")
    builder.make_ramp_course()
    builder.add_pushable_box(0, 2.8, id="crate_box")
    builder.add_cylinder(1.4, 2.8, id="barrel_cylinder")
    builder.add_ball(-1.4, 2.8, id="round_ball")
    builder.add_hazard_zone(3.2, 2.8, id="hazard")
    return builder.finalize()


def test_visual_scene_maps_supported_semantics() -> None:
    scene = compile_visual_scene(_coverage_spec())
    by_source = {
        item["source_id"]: item
        for item in scene["objects"]
        if item["physics_backed"] and item.get("source_id")
    }

    assert scene["theme"] == DEFAULT_VISUAL_THEME
    assert by_source["ground"]["visual_type"] == "terrain_grass"
    assert by_source["main_ramp"]["visual_type"] == "wood_ramp"
    assert by_source["upper_platform"]["visual_type"] == "wood_platform"
    assert by_source["crate_box"]["visual_type"] == "crate"
    assert by_source["barrel_cylinder"]["visual_type"] == "barrel"
    assert by_source["round_ball"]["visual_type"] == "boulder"
    assert by_source["agent"]["visual_type"] == "agent_hero"
    assert by_source["goal"]["visual_type"] == "goal_portal"
    assert by_source["hazard"]["visual_type"] == "hazard_spikes"
    assert scene["environment"]["sky"]["style"] == "storybook_sky"
    assert scene["environment"]["set_dressing"]["placement"] == "perimeter_only"
    assert scene["environment"]["set_dressing"]["grounding"] == "continuous_meadow"
    assert any(item["visual_type"] == "tree" for item in scene["objects"])
    assert not any(item["visual_type"] == "path_tile" for item in scene["objects"])


def test_visual_decorations_stay_outside_primary_play_area() -> None:
    spec = _coverage_spec()
    scene = compile_visual_scene(spec)
    ground = next(item for item in scene["objects"] if item["source_id"] == "ground")
    center_x, center_y, _ = ground["position"]
    half_width = ground["size"][0] * 0.5
    half_depth = ground["size"][1] * 0.5

    for item in scene["objects"]:
        if item["physics_backed"]:
            continue
        x, y, _ = item["position"]
        inside_play_area = (
            center_x - half_width <= x <= center_x + half_width
            and center_y - half_depth <= y <= center_y + half_depth
        )
        assert not inside_play_area, item["id"]


def test_groundless_scene_does_not_add_floating_decorations() -> None:
    builder = EnvSpec3DBuilder("groundless_visual", description="a suspended wall")
    builder.add_wall(0.0, 0.0, 4.0, 0.4, 1.0, id="wall")

    scene = compile_visual_scene(builder.finalize())

    assert [item["source_id"] for item in scene["objects"]] == ["wall"]
    assert all(item["physics_backed"] for item in scene["objects"])


def test_visual_scene_is_deterministic() -> None:
    spec = _coverage_spec()

    assert compile_visual_scene(spec) == compile_visual_scene(spec)


def test_visual_only_decorations_do_not_enter_mjcf() -> None:
    spec = _coverage_spec()
    visual_scene = compile_visual_scene(spec)
    xml = compile_spec_to_mjcf(spec)
    root = ET.fromstring(xml)
    geom_names = {geom.attrib["name"] for geom in root.findall(".//geom")}
    body_names = {body.attrib["name"] for body in root.findall(".//body")}

    visual_only_ids = {
        item["id"]
        for item in visual_scene["objects"]
        if not item["physics_backed"]
    }
    assert visual_only_ids
    assert visual_only_ids.isdisjoint(geom_names)
    assert visual_only_ids.isdisjoint(body_names)


def test_visual_scene_artifact_and_legacy_load(tmp_path) -> None:
    spec = _coverage_spec()
    result = persist_artifacts(
        spec=spec,
        scene_dir=tmp_path / spec.id,
        trace_records=[],
        render=False,
        display_name="Stacked Boxes",
    )
    visual_path = tmp_path / spec.id / VISUAL_SCENE_FILENAME

    assert visual_path.is_file()
    assert "visual_scene" in result["metadata"]["artifacts"]
    loaded = load_scene(tmp_path / spec.id)
    assert loaded is not None
    assert loaded["display_name"] == "Stacked Boxes"
    assert loaded["visual_scene"]["theme"] == DEFAULT_VISUAL_THEME
    assert loaded["visual_scene_url"].endswith(f"/{VISUAL_SCENE_FILENAME}?v={visual_path.stat().st_mtime_ns}")

    visual_path.unlink()
    legacy = load_scene(tmp_path / spec.id)
    assert legacy is not None
    assert legacy["visual_scene"]["source_env_id"] == spec.id
    assert legacy["visual_scene_url"] is None

    persist_artifacts(spec=spec, scene_dir=tmp_path / spec.id, trace_records=[], render=False)
    assert load_scene(tmp_path / spec.id)["display_name"] == "Stacked Boxes"


def test_visual_scene_artifact_can_be_refreshed_without_rewriting_authored_state(tmp_path) -> None:
    spec = _coverage_spec()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    spec_path = scene_dir / "env_spec_3d.json"
    original_spec = spec_path.read_bytes()
    visual_path = scene_dir / VISUAL_SCENE_FILENAME
    visual_path.write_text('{"stale": true}\n', encoding="utf-8")

    refreshed_path = refresh_visual_scene_artifact(scene_dir)

    assert refreshed_path == visual_path
    assert json.loads(visual_path.read_text())["source_env_id"] == spec.id
    assert spec_path.read_bytes() == original_spec
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    assert metadata["artifacts"]["visual_scene"]["role"] == "derived_visual_scene"


def test_load_scene_exposes_env_verification_artifacts(tmp_path) -> None:
    spec = _coverage_spec()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    spec_json = spec.model_dump(mode="json")
    plan = normalize_env_verification_plan(
        env_id=spec.id,
        prompt="needs an agent",
        checks=[{"id": "one_agent", "type": "object_count", "selector": "agent", "exact": 1}],
        operation_count=0,
        draft_spec=spec_json,
    )
    report = run_env_verification(
        env_id=spec.id,
        plan=plan,
        draft_spec=spec_json,
        final_spec=spec_json,
        operation_count=0,
    )
    write_env_verification_plan(scene_dir, plan)
    write_env_verification_report(scene_dir, report)

    loaded = load_scene(scene_dir)

    assert loaded is not None
    assert loaded["env_verification"]["status"] == "passed"
    assert loaded["env_verification_plan"]["checks"][0]["id"] == "one_agent"
    assert loaded["env_verification_report"]["status"] == "passed"


def test_load_scene_preserves_public_history_activity(tmp_path) -> None:
    spec = _coverage_spec()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    (scene_dir / HISTORY_FILENAME).write_text(
        json.dumps(
            {
                "env_id": spec.id,
                "turns": [
                    {"role": "user", "content": "make a box"},
                    {
                        "role": "assistant",
                        "content": "Created and finalized the environment.",
                        "activity": [
                            {"type": "agent_message", "label": "Agent message", "message": "I added the requested box."},
                            {"type": "reasoning", "label": "Reasoning", "message": "hidden"},
                            {"type": "mcp_tool_call", "label": "Tool call", "message": "{'env_id': 'demo'}"},
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_scene(scene_dir)

    assert loaded is not None
    assert loaded["history"][1]["activity"] == [
        {"type": "agent_message", "label": "Agent message", "message": "I added the requested box."},
        {"type": "mcp_tool_call", "label": "Tool call", "message": "{'env_id': 'demo'}"},
    ]


def test_load_scene_exposes_visual_review_defaults_and_hashes(tmp_path) -> None:
    spec = _coverage_spec()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)

    loaded = load_scene(scene_dir)

    assert loaded is not None
    assert loaded["spec_hash"]
    assert loaded["visual_scene_hash"]
    assert loaded["env_visual_review"]["status"] == "not_run"
    assert loaded["env_visual_review_report"] is None
    assert loaded["env_visual_review_pending"] is None
    assert loaded["env_visual_review_history"] == []
