from __future__ import annotations

import base64
import json

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.mcp_server import EnvironmentGenerationMCPServer
from environment_generation.studio_view_context import persist_studio_view_context
from environment_generation.task_agent import TASK_AGENT_OBSERVATION_MODE


def test_mcp_tools_list_contains_generation_tools(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    initialized = server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
    assert initialized["result"]["serverInfo"]["name"] == "environment-generation"

    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    names = set(tools)

    assert {
        "create_scene",
        "resume_scene",
        "list_object_catalog",
        "apply_operation",
        "inspect_scene",
        "render_scene_preview",
        "validate_scene",
        "define_env_verification_plan",
        "run_env_verification",
        "get_env_verification_report",
        "define_env_behavior_trials",
        "preserve_env_behavior_trials",
        "use_default_env_behavior_trial",
        "get_env_behavior_trial_report",
        "finalize_scene",
    } <= names

    operation_names = tools["apply_operation"]["inputSchema"]["properties"]["operation"]["properties"]["op"]["enum"]
    assert "configure_reach_goal_game" in operation_names
    assert "set_ramp_geometry" in operation_names
    operation_variants = tools["apply_operation"]["inputSchema"]["properties"]["operation"]["oneOf"]
    set_ramp_variant = next(
        variant for variant in operation_variants if variant["properties"]["op"]["const"] == "set_ramp_geometry"
    )
    assert "rise" in set_ramp_variant["properties"]["args"]["properties"]
    assert "height" not in set_ramp_variant["properties"]["args"]["properties"]
    check_schema = tools["define_env_verification_plan"]["inputSchema"]["properties"]["checks"]["items"]
    assert "support_contact" in check_schema["properties"]["type"]["enum"]
    assert "ramp_connection" in check_schema["properties"]["type"]["enum"]
    assert "screen_region" in check_schema["properties"]["type"]["enum"]
    assert "screen_relation" in check_schema["properties"]["type"]["enum"]
    assert "bottom_left" in check_schema["properties"]["region"]["enum"]
    assert "pushable_moves" in check_schema["properties"]["probe"]["enum"]
    behavior_trials = tools["define_env_behavior_trials"]["inputSchema"]["properties"]["trials"]
    assert "intent_summary" in tools["define_env_behavior_trials"]["inputSchema"]["required"]
    assert tools["preserve_env_behavior_trials"]["inputSchema"]["required"] == ["env_id", "reason"]
    assert tools["use_default_env_behavior_trial"]["inputSchema"]["required"] == ["env_id", "reason"]
    assert behavior_trials["maxItems"] == 2
    behavior_check = behavior_trials["items"]["properties"]["objective"]["properties"]["checks"]["items"]
    assert "description" in behavior_check["required"]
    assert behavior_check["properties"]["description"]["minLength"] == 8
    assert "subject" in behavior_check["properties"]["description"]["description"]
    behavior_predicate = behavior_check["properties"]["predicate"]
    behavior_check_types = behavior_predicate["properties"]["type"]["enum"]
    assert "relation" in behavior_check_types
    assert "overlap" in behavior_check_types
    assert "displacement" in behavior_check_types
    assert "terminal_event" in behavior_check_types
    assert "reset_event" in behavior_check_types
    assert set(behavior_predicate["properties"]["event"]["enum"]) == {
        "goal",
        "hazard",
        "out_of_bounds",
    }
    assert "any" in behavior_predicate["properties"]["reason"]["enum"]
    assert behavior_predicate["properties"]["subject"] == behavior_predicate["properties"]["target"]


def test_object_catalog_exposes_operation_specific_usage(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_object_catalog", "arguments": {}},
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    operations = {operation["name"]: operation for operation in payload["operations"]}

    assert "low walkable endpoint" in operations["set_ramp_geometry"]["description"]
    assert "rise" in operations["set_ramp_geometry"]["arguments"]["properties"]
    assert "height" not in operations["set_ramp_geometry"]["arguments"]["properties"]


def test_bound_behavior_mcp_exposes_navigation_only(tmp_path, monkeypatch) -> None:
    builder = EnvSpec3DBuilder("bound_behavior", description="bound behavior")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = {
        "id": "move",
        "instruction": "Move one meter.",
        "expected_outcome": "should_succeed",
        "severity": "critical",
        "max_steps": 200,
        "max_resets": 0,
        "objective": {
            "mode": "all",
            "checks": [
                {
                    "id": "move",
                    "type": "agent_displacement",
                    "min_distance": 1.0,
                    "space": "xy",
                    "description": "",
                }
            ],
            "ordered_check_ids": [],
        },
    }
    monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_SCENE_DIR", str(scene_dir))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_TRIAL_JSON", json.dumps(trial))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_ACTION_LOG", str(tmp_path / "actions.jsonl"))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_FRAME_DIR", str(tmp_path / "frames"))

    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    names = set(tools)

    assert names == {
        "start_behavior_trial",
        "observe_behavior_trial",
        "act_behavior_trial",
        "reset_behavior_trial",
        "stop_behavior_trial",
    }
    assert "apply_operation" not in names
    act_schema = tools["act_behavior_trial"]["inputSchema"]["properties"]
    assert act_schema["assist"]["enum"] == ["none", "ground_route"]
    assert "intermediate target" in act_schema["target_id"]["description"]
    assert act_schema["frames"]["maximum"] > 60
    server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "start_behavior_trial", "arguments": {}},
        }
    )
    stopped = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "stop_behavior_trial", "arguments": {}},
        }
    )
    stopped_payload = json.loads(stopped["result"]["content"][0]["text"])
    assert "actions" not in stopped_payload
    assert "trajectory" not in stopped_payload
    assert stopped_payload["action_count"] == 0
    assert stopped_payload["trajectory_frame_count"] >= 1
    server.behavior_session.close()


def test_bound_task_mcp_exposes_task_controller_only(tmp_path, monkeypatch) -> None:
    builder = EnvSpec3DBuilder("bound_task", description="bound task")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0, id="robot")
    builder.add_goal_zone(2, 0, id="goal")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    task = {
        "task_id": "reach",
        "env_id": spec.id,
        "instruction": "Reach the goal.",
        "max_steps": 200,
        "tests": [
            {
                "id": "reach",
                "description": "Reach the goal.",
                "mode": "all",
                "source": "codex",
                "ordered_condition_ids": [],
                "conditions": [
                    {
                        "id": "enter",
                        "description": "Enter goal.",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "robot"},
                            "target": {"id": "goal"},
                        },
                    }
                ],
            }
        ],
    }
    monkeypatch.setenv("ENVIRONMENT_GENERATION_TASK_SCENE_DIR", str(scene_dir))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_TASK_JSON", json.dumps(task))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_TASK_ACTION_LOG", str(tmp_path / "task-actions.jsonl"))
    monkeypatch.setenv("ENVIRONMENT_GENERATION_TASK_FRAME_DIR", str(tmp_path / "task-frames"))

    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    names = set(tools)

    assert names == {
        "start_task_run",
        "observe_task_run",
        "act_task_run",
        "reset_task_run",
        "stop_task_run",
    }
    assert "apply_operation" not in names
    assert "define_env_verification_plan" not in names
    action_frames = tools["act_task_run"]["inputSchema"]["properties"]["frames"]
    assert action_frames["maximum"] == 120
    assert action_frames["default"] == 24
    assert "Earlier views remain available" in tools["observe_task_run"]["description"]

    initialized = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "initialize"})
    instructions = " ".join(initialized["result"]["instructions"].split())
    assert "visual-first" in instructions
    assert "does not expose coordinates" in instructions
    assert "anonymous collision" in instructions
    assert "global preview" in instructions
    assert "live deterministic test\nresults" not in instructions

    started = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "start_task_run", "arguments": {}},
        }
    )
    observation = json.loads(started["result"]["content"][0]["text"])
    assert observation["observation_mode"] == TASK_AGENT_OBSERVATION_MODE
    assert isinstance(observation["grounded"], bool)
    assert isinstance(observation["collision"], bool)
    assert observation["recent_events"] == []
    assert observation["recent_actions"] == []
    assert len(observation["recent_frames"]) <= 1
    assert observation["recent_frames_order"] == "oldest_to_current"
    assert {
        "agent",
        "camera",
        "mechanisms",
        "navigation",
        "nearby_objects",
        "tests",
    }.isdisjoint(observation)
    assert all(
        key not in json.dumps(observation)
        for key in ("zone_id", "semantic_type", "subject_id", "position", "target_bearing")
    )

    stopped = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "stop_task_run", "arguments": {}},
        }
    )
    final_result = json.loads(stopped["result"]["content"][0]["text"])
    assert final_result["observation_mode"] == TASK_AGENT_OBSERVATION_MODE
    assert "report" not in final_result
    assert "termination_reason" not in final_result
    server.task_session.close()


def test_task_result_attaches_bounded_recent_frames_oldest_to_current(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    current = tmp_path / "current.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    current.write_bytes(b"current")

    content = server._content_for_result(
        {
            "status": "success",
            "recent_frames": [
                {"path": str(first), "step": 0},
                {"path": str(second), "step": 8},
                {"path": str(current), "step": 16},
            ],
            "path": str(current),
        }
    )

    assert [item["type"] for item in content] == ["text", "image", "image", "image"]
    assert [base64.b64decode(item["data"]) for item in content[1:]] == [
        b"first",
        b"second",
        b"current",
    ]
    assert len([item for item in content if item["type"] == "text"]) == 1


def test_task_result_attaches_only_current_frame_without_repeating_visual_history(
    tmp_path,
) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    previous = tmp_path / "previous.png"
    current = tmp_path / "current.png"
    previous.write_bytes(b"previous")
    current.write_bytes(b"current")

    content = server._content_for_result(
        {
            "status": "success",
            "observation_mode": TASK_AGENT_OBSERVATION_MODE,
            "recent_frames": [
                {"path": str(previous), "step": 0},
                {"path": str(current), "step": 24},
            ],
            "path": str(current),
        }
    )

    assert [item["type"] for item in content] == ["text", "image"]
    assert base64.b64decode(content[1]["data"]) == b"current"


def test_mcp_create_and_apply_operation(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    create = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "create_scene", "arguments": {"env_id": "scene_1", "prompt": "box scene with an agent and goal"}},
        }
    )
    apply = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "apply_operation",
                "arguments": {"env_id": "scene_1", "operation": {"op": "make_box_goal_scene", "args": {}}},
            },
        }
    )

    assert "scene_1" in create["result"]["content"][0]["text"]
    assert '"status": "success"' in apply["result"]["content"][0]["text"]


def test_mcp_resume_scene_and_apply_targeted_edit(tmp_path) -> None:
    builder = EnvSpec3DBuilder("saved_scene", description="saved")
    builder.make_box_goal_scene()
    spec = builder.finalize()
    persist_artifacts(spec=spec, scene_dir=tmp_path / spec.id, trace_records=[], render=False)
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    resume = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "resume_scene", "arguments": {"env_id": "saved_scene"}},
        }
    )
    move = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "apply_operation",
                "arguments": {
                    "env_id": "saved_scene",
                    "operation": {"op": "move_object", "args": {"id": "goal", "x": 5.0}},
                },
            },
        }
    )

    resume_payload = json.loads(resume["result"]["content"][0]["text"])
    move_payload = json.loads(move["result"]["content"][0]["text"])
    goal = next(obj for obj in move_payload["draft_summary"]["objects"] if obj["id"] == "goal")
    assert resume_payload["status"] == "success"
    assert move_payload["status"] == "success"
    assert goal["position"][0] == 5.0


def test_mcp_env_verification_tools_persist_plan_and_report(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    _call(
        server,
        1,
        "create_scene",
        {"env_id": "verified_scene", "prompt": "box scene with an agent and goal"},
    )
    _call(
        server,
        2,
        "apply_operation",
        {"env_id": "verified_scene", "operation": {"op": "make_box_goal_scene", "args": {}}},
    )
    define = _call(
        server,
        3,
        "define_env_verification_plan",
        {
            "env_id": "verified_scene",
            "checks": [
                {"id": "one_agent", "type": "object_count", "selector": "agent", "exact": 1},
                {"id": "box_supported", "type": "support_contact", "selector": "pushable_box"},
            ],
        },
    )
    run = _call(server, 4, "run_env_verification", {"env_id": "verified_scene"})
    report = _call(server, 5, "get_env_verification_report", {"env_id": "verified_scene"})

    assert define["status"] == "success"
    assert run["status"] == "success"
    assert report["plan"]["checks"][0]["id"] == "one_agent"
    assert report["report"]["status"] == "passed"
    assert (tmp_path / "verified_scene" / "env_verification_plan.json").is_file()
    assert (tmp_path / "verified_scene" / "env_verification_report.json").is_file()


def test_studio_screen_requirement_cannot_be_dropped_from_verification_plan(tmp_path) -> None:
    builder = EnvSpec3DBuilder("screen_scene", description="screen scene")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "screen_scene"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    prompt = "move the agent to the bottom left"
    persist_studio_view_context(
        scene_dir=scene_dir,
        review_id="turn-screen",
        prompt=prompt,
        view_context={
            "screen_space": {
                "camera": {
                    "position": [3.4, 11.8, 9.5],
                    "target": [0, 0, 0.5],
                    "fov_y_degrees": 45,
                    "aspect": 16 / 9,
                },
                "regions": {
                    "bottom_left": {
                        "bounds_uv": {"left": 0.05, "top": 0.58, "right": 0.34, "bottom": 0.9},
                        "anchor": {"world_position": [-5, 5, 0]},
                    }
                },
            }
        },
    )
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    _call(server, 1, "resume_scene", {"env_id": "screen_scene", "prompt": prompt})

    rejected = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "define_env_verification_plan",
                "arguments": {
                    "env_id": "screen_scene",
                    "checks": [{"type": "object_count", "selector": "agent", "exact": 1}],
                },
            },
        }
    )
    assert "requires a screen_region check for bottom-left" in rejected["error"]["message"]

    accepted = _call(
        server,
        3,
        "define_env_verification_plan",
        {
            "env_id": "screen_scene",
            "checks": [
                {
                    "id": "agent_submitted_bottom_left",
                    "type": "screen_region",
                    "subject": {"id": "agent"},
                    "region": "bottom_left",
                }
            ],
        },
    )
    assert accepted["plan"]["checks"][0]["projection"]["context_review_id"] == "turn-screen"


def test_mcp_behavior_tools_persist_typed_plan(tmp_path) -> None:
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)
    _call(server, 1, "create_scene", {"env_id": "behavior_scene", "prompt": "agent climbs the box on the way to a goal"})
    _call(
        server,
        2,
        "apply_operation",
        {"env_id": "behavior_scene", "operation": {"op": "make_box_goal_scene", "args": {}}},
    )

    defined = _call(
        server,
        3,
        "define_env_behavior_trials",
        {
            "env_id": "behavior_scene",
            "intent_summary": "Demonstrate the requested box-climbing affordance.",
            "trials": [
                {
                    "id": "climb_box",
                    "instruction": "Jump onto the pushable box.",
                    "objective": {
                        "checks": [
                            {"id": "jump", "type": "jump_count", "min_count": 1},
                            {
                                "id": "contact",
                                "type": "contact_count",
                                "selector": {"id": "pushable_box"},
                            },
                        ],
                        "ordered_check_ids": ["jump", "contact"],
                    },
                }
            ],
        },
    )
    loaded = _call(server, 4, "get_env_behavior_trial_report", {"env_id": "behavior_scene"})

    assert defined["status"] == "success"
    assert defined["plan"]["decision"] == "prompt_specific"
    assert defined["plan"]["intent_summary"] == "Demonstrate the requested box-climbing affordance."
    assert loaded["plan"]["trials"][0]["id"] == "climb_box"
    assert loaded["report"] is None
    assert (tmp_path / "behavior_scene" / "env_behavior_trials_plan.json").is_file()


def test_mcp_finalization_blocks_on_missing_and_failed_critical_env_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist_artifacts)
    server = EnvironmentGenerationMCPServer(output_root=tmp_path)

    _call(server, 1, "create_scene", {"env_id": "blocked_scene", "prompt": "box scene with an agent and goal"})
    _call(
        server,
        2,
        "apply_operation",
        {"env_id": "blocked_scene", "operation": {"op": "make_box_goal_scene", "args": {}}},
    )
    _call(
        server,
        3,
        "define_env_verification_plan",
        {
            "env_id": "blocked_scene",
            "checks": [{"id": "two_goals", "type": "object_count", "selector": "goal", "exact": 2}],
        },
    )

    missing = _call(server, 4, "finalize_scene", {"env_id": "blocked_scene"})
    failed = _call(server, 5, "run_env_verification", {"env_id": "blocked_scene"})
    blocked = _call(server, 6, "finalize_scene", {"env_id": "blocked_scene"})
    _call(
        server,
        7,
        "define_env_verification_plan",
        {
            "env_id": "blocked_scene",
            "checks": [{"id": "one_goal", "type": "object_count", "selector": "goal", "exact": 1}],
        },
    )
    passed = _call(server, 8, "run_env_verification", {"env_id": "blocked_scene"})
    behavior_blocked = _call(server, 9, "finalize_scene", {"env_id": "blocked_scene"})
    selected = _call(
        server,
        10,
        "use_default_env_behavior_trial",
        {
            "env_id": "blocked_scene",
            "reason": "The test fixture has no prompt-specific behavior beyond its standard game contract.",
        },
    )
    finalized = _call(server, 11, "finalize_scene", {"env_id": "blocked_scene"})

    assert missing["status"] == "needs_changes"
    assert missing["env_verification"]["reason"] == "missing_report"
    assert failed["status"] == "needs_changes"
    assert blocked["env_verification"]["reason"] == "critical_failures"
    assert passed["status"] == "success"
    assert behavior_blocked["env_behavior_trials"]["status"] == "needs_decision"
    assert selected["plan"]["decision"] == "fallback"
    assert finalized["status"] == "success"


def _call(server: EnvironmentGenerationMCPServer, request_id: int, name: str, arguments: dict) -> dict:
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return json.loads(response["result"]["content"][0]["text"])


def _fake_persist_artifacts(*, spec, scene_dir, trace_records, render=True):
    return {
        "paths": {
            "env_spec": str(scene_dir / "env_spec_3d.json"),
            "visual_scene": str(scene_dir / "visual_scene.json"),
            "world_xml": str(scene_dir / "world.xml"),
            "metadata": str(scene_dir / "metadata.json"),
            "trace": str(scene_dir / "generation_trace.jsonl"),
            "previews": {},
            "orbit_previews": [],
        },
        "metadata": {"env_id": spec.id},
    }
